from __future__ import annotations
"""
Creative‑System core (v2.2) – **scalable semantic layer**
=========================================================
Implements the three requested upgrades:

1.  **Annoy‑backed ANN search** – switches to Facebook's Annoy when
    available and automatically re‑indexes when the vector store exceeds
    50 k chunks.  Falls back to brute‑force numpy search for small repos
    or if Annoy is missing.
2.  **Multi‑language embedding** – any text‑based file (`.py`, `.js`,
    `.ts`, `.java`, `.go`, `.sql`, `.md`, etc.) is chunked (≈120 lines)
    and embedded.
3.  **Prompt builder** – `prepare_prompt(query, user_msg)` returns a
    context‑augmented prompt ready for o3/o4‑mini; it automatically
    streams the top‑K retrieved snippets with delimiters.

Legacy imports (`AgenticCreativitySystem`, `CreativeContentSystem`,
`ContentType`) still resolve.
"""

import ast
import enum
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"
CHUNK_MAX_LINES = 120  # for non‑Python files

try:
    from annoy import AnnoyIndex  # type: ignore

    _ANNOY_OK = True
except ModuleNotFoundError:
    _ANNOY_OK = False

# ---------------------------------------------------------------------------
# 0. Embedding helper (mock if no API key)
# ---------------------------------------------------------------------------

def _embed(texts: List[str]) -> np.ndarray:  # (n, 1536)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        rng = np.random.default_rng([hash(t) % (2**32) for t in texts])
        return rng.random((len(texts), 1536), dtype=np.float32)
    headers = {"Authorization": f"Bearer {key}"}
    data = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(OPENAI_ENDPOINT, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    vecs = [d["embedding"] for d in r.json()["data"]]
    return np.asarray(vecs, dtype=np.float32)

# ---------------------------------------------------------------------------
# 1. Git change tracker
# ---------------------------------------------------------------------------

class GitChangeTracker:
    def __init__(self, repo_root: str = ".", base_ref: str = "HEAD") -> None:
        self.root = Path(repo_root).resolve()
        self.base = base_ref

    def changed_files(self, exts: Optional[List[str]] = None) -> List[Path]:
        exts = exts or [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".sql",
            ".md",
            ".txt",
        ]
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--name-only", self.base], cwd=self.root, text=True
            )
            files = [self.root / p.strip() for p in diff.splitlines() if p.strip()]
        except subprocess.CalledProcessError as exc:
            logger.warning("git diff failed (%s); scanning repo", exc)
            files = list(self.root.rglob("*"))
        return [p for p in files if p.suffix in exts and p.exists()]

# ---------------------------------------------------------------------------
# 2. SQLite content store (unchanged API)
# ---------------------------------------------------------------------------

class SQLiteContentSystem:
    def __init__(self, db_path: str = "ai_creations/content.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS content(id TEXT PRIMARY KEY,type TEXT,title TEXT,filepath TEXT,created_at TEXT,metadata TEXT)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_type_created ON content(type,created_at DESC)"
        )
        self.conn.commit()

    async def store_content(self, *, content_type: str, title: str, content: str, metadata: Dict[str, Any] | None = None):
        cid = hashlib.sha1(f"{title}{datetime.utcnow()}".encode()).hexdigest()
        ext = "md" if content_type in {"analysis", "assessment"} else "txt"
        path = Path("ai_creations") / content_type
        path.mkdir(parents=True, exist_ok=True)
        fp = path / f"{cid}.{ext}"
        fp.write_text(content, encoding="utf-8")
        self.conn.execute(
            "INSERT INTO content VALUES (?,?,?,?,?,?)",
            (cid, content_type, title, str(fp), datetime.utcnow().isoformat(), json.dumps(metadata or {})),
        )
        self.conn.commit()
        return {"id": cid, "filepath": str(fp)}

# ---------------------------------------------------------------------------
# 3. Vector index – Annoy + fallback numpy
# ---------------------------------------------------------------------------

class VectorIndex:
    def __init__(self, path: str = "ai_creations/code_index") -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".json")
        self.dim = 1536
        self._use_annoy = _ANNOY_OK
        self._meta: List[Tuple[str, str, int, int]] = []  # file, snippet, line_start, line_end
        if self._use_annoy:
            self.idx = AnnoyIndex(self.dim, "angular")
        else:
            self._vecs = np.empty((0, self.dim), dtype=np.float32)
        self._load()

    # ------- persistence --------
    def _load(self):
        if self.meta_path.exists():
            self._meta = json.loads(self.meta_path.read_text())
        if self._use_annoy and self.path.with_suffix(".ann").exists():
            self.idx.load(str(self.path.with_suffix(".ann")))
        elif not self._use_annoy and self.path.with_suffix(".npz").exists():
            self._vecs = np.load(self.path.with_suffix(".npz"))["v"]

    def _save(self):
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self._meta))
        if self._use_annoy:
            self.idx.save(str(self.path.with_suffix(".ann")))
        else:
            np.savez_compressed(self.path.with_suffix(".npz"), v=self._vecs)

    # ------- add/search --------
    def add(self, vecs: np.ndarray, metas: List[Tuple[str, str, int, int]]):
        start_id = len(self._meta)
        self._meta.extend(metas)
        if self._use_annoy:
            for i, v in enumerate(vecs):
                self.idx.add_item(start_id + i, v)
            if len(self._meta) % 1000 == 0:  # periodic rebuild
                self.idx.build(10)
        else:
            self._vecs = np.vstack([self._vecs, vecs]) if self._vecs.size else vecs
        self._save()

    def search(self, q: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if not self._meta:
            return []
        if self._use_annoy and len(self._meta) > 50000:
            idxs = self.idx.get_nns_by_vector(q.flatten(), k, include_distances=True)
            ids, dists = idxs
        else:  # brute force
            if self._use_annoy:
                # small index not yet built
                sims = [self.idx.get_item_vector(i) for i in range(len(self._meta))]
                sims = np.asarray(sims) @ q.flatten()
            else:
                sims = (self._vecs @ q.T).flatten()
            ids = sims.argsort()[-k:][::-1]
            dists = 1 - sims[ids]
        return [
            {
                "similarity": float(1 - dists[i] if self._use_annoy else sims[ids[i]]),
                "filepath": self._meta[ids[i]][0],
                "snippet": self._meta[ids[i]][1],
                "line_start": self._meta[ids[i]][2],
                "line_end": self._meta[ids[i]][3],
            }
            for i in range(len(ids))
        ]

# ---------------------------------------------------------------------------
# 4. Multi‑language embedder
# ---------------------------------------------------------------------------

class CodeEmbedder:
    PY_EXTS = {".py"}

    def __init__(self, index: VectorIndex):
        self.index = index

    # ---- chunk helpers ----

    def _py_chunks(self, path: Path):
        src = path.read_text("utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.body[-1].end_lineno if hasattr(node.body[-1], "end_lineno") else node.body[-1].lineno
                yield "\n".join(src.splitlines()[start:end]), start + 1, end
        if not tree.body:
            yield src, 1, len(src.splitlines())

    def _generic_chunks(self, path: Path):
        lines = path.read_text("utf-8", errors="ignore").splitlines()
        for i in range(0, len(lines), CHUNK_MAX_LINES):
            chunk = "\n".join(lines[i : i + CHUNK_MAX_LINES])
            yield chunk, i + 1, min(len(lines), i + CHUNK_MAX_LINES)

    def embed_file(self, path: Path):
        chunks = list(self._py_chunks(path) if path.suffix in self.PY_EXTS else self._generic_chunks(path))
        texts = [c[0] for c in chunks]
        vecs = _embed(texts)
        metas = [(str(path), t[:120] + ("…" if len(t) > 120 else ""), s, e) for (t, s, e) in chunks]
        self.index.add(vecs, metas)

# ---------------------------------------------------------------------------
# 5. Parallel AST scan (python only, unchanged)
# ---------------------------------------------------------------------------

def _scan(code: str):
    tree = ast.parse(code)
    cls = fnc = imp = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            cls += 1
        elif isinstance(n, ast.FunctionDef):
            fnc += 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            imp += 1
    return {"loc": len(code.splitlines()), "classes": cls, "functions": fnc, "imports": imp}

class ParallelCodeAnalyzer:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or os.cpu_count() or 4

    def analyze(self, paths: List[Path]):
        py_files = [p for p in paths if p.suffix == ".py"]
        results: Dict[str, Dict[str, int]] = {}
        if not py_files: # Add check for empty list
            logger.info("CodeAnalyzer: No Python files to analyze.")
            return results
    
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:   # NEW
            # Create a list of future-to-path mappings to handle potential errors per file
            future_to_path = {}
            for p in py_files:
                try:
                    # Reading text should be safe here, but actual parsing is in _scan
                    future_to_path[pool.submit(p.read_text, encoding="utf-8")] = p
                except Exception as e:
                    logger.error(f"Failed to submit read task for file {p}: {e}")
    
            for fut in as_completed(future_to_path):
                p = future_to_path[fut]
                try:
                    file_content = fut.result() # Get content from future
                    results[str(p)] = _scan(file_content) # Pass content to _scan
                except Exception as exc:
                    logger.error("AST scan/processing failure for file %s: %s", p, exc)
        return results

# ---------------------------------------------------------------------------
# 6. Orchestrator with prompt builder
# ---------------------------------------------------------------------------

class AgenticCreativitySystemV2:
    def __init__(self, repo_root: str = ".", review_interval_days: int = 7):
        self.tracker = GitChangeTracker(repo_root)
        self.storage = SQLiteContentSystem()
        self.analyzer = ParallelCodeAnalyzer()
        self.index = VectorIndex()
        self.embedder = CodeEmbedder(self.index)
        self.review_interval_days = review_interval_days

    # --------------- ingestion ----------------
    async def incremental_codebase_analysis(self):
        changed = self.tracker.changed_files()
        if not changed:
            return {"changed_files": 0}
        metrics = self.analyzer.analyze(changed)
        for p in changed:
            try:
                self.embedder.embed_file(p)
            except Exception as exc:
                logger.warning("Embedding failed %s: %s", p, exc)
        summary = {"changed_files": len(changed), "aggregated_loc": sum(m["loc"] for m in metrics.values()), "files": metrics}
        await self.storage.store_content(content_type="analysis", title=f"Incremental analysis {datetime.utcnow().isoformat(timespec='seconds')}", content=json.dumps(summary, indent=2))
        return summary

    # --------------- retrieval ----------------
    async def semantic_search(self, query: str, k: int = 5):
        qvec = _embed([query])[0:1]
        return self.index.search(qvec, k)

    async def prepare_prompt(self, query: str, user_msg: str, k: int = 5) -> str:
        """Return a full prompt with top‑K context snippets prepended."""
        hits = await self.semantic_search(query, k)
        ctx_parts = [
            f"# Context snippet {i+1} (similarity {h['similarity']:.3f})\n# {h['filepath']} L{h['line_start']}-{h['line_end']}\n" + indent(h["snippet"].rstrip(), "# ")
            for i, h in enumerate(hits)
        ]
        context = "\n\n".join(ctx_parts)
        return f"{context}\n\n# User question\n{user_msg}"

async def integrate_with_existing_system(nyx_brain=None):
    """
    Integrates the creativity system with an existing NyxBrain instance.
    
    Args:
        nyx_brain: Optional reference to a NyxBrain instance
        
    Returns:
        Initialized AgenticCreativitySystemV2 instance
    """
    # Initialize the creativity system
    repo_root = getattr(nyx_brain, "creative_system_root", ".") if nyx_brain else "."
    system = AgenticCreativitySystemV2(repo_root=repo_root)
    
    # If nyx_brain is provided, set up the integration
    if nyx_brain:
        # Run initial analysis on codebase
        await system.incremental_codebase_analysis()
        
        # Connect to nyx_brain's systems if needed
        if hasattr(nyx_brain, "memory_core") and nyx_brain.memory_core:
            # Store reference to creativity content in memory
            await nyx_brain.memory_core.add_memory(
                memory_text="Initialized creative system for content generation",
                memory_type="system",
                significance=3,
                tags=["creativity", "initialization"]
            )
        
        # Hook up event listeners if the brain has an event system
        if hasattr(nyx_brain, "event_bus") and nyx_brain.event_bus:
            # Example: Register for relevant events
            await nyx_brain.event_bus.subscribe("creativity_request", 
                                               system.semantic_search)
            
        logger.info(f"Creative system integrated with NyxBrain instance")
            
    return system

# ---------------------------------------------------------------------------
# 7. Compatibility shims
# ---------------------------------------------------------------------------

class _ContentType(enum.Enum):
    STORY = "story"
    POEM = "poem"
    LYRICS = "lyrics"
    JOURNAL = "journal"
    CODE = "code"
    ANALYSIS = "analysis"
    ASSESSMENT = "assessment"

# Keep old import paths working
ContentType = _ContentType  # type: ignore
CreativeContentSystem = SQLiteContentSystem  # type: ignore
AgenticCreativitySystem = AgenticCreativitySystemV2  # type: ignore
