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
    def __init__(self, repo_root: str = ".", base_ref: Optional[str] = None) -> None:
        self.root = Path(repo_root).resolve()
        self.base_ref_for_diff = base_ref
        self.is_git_repo = (self.root / ".git").is_dir()
        self.initial_run_done = False # To track if the first run with list_all_tracked has occurred

        if not self.is_git_repo:
            logger.warning(
                f"Directory '{self.root}' does not appear to be a Git repository (no .git folder found). "
                f"Git-based change tracking will be effectively disabled for this tracker instance. "
                f"The system will assume no files have changed according to git for incremental analysis."
            )
        else:
            logger.info(f"GitChangeTracker initialized for git repo at '{self.root}'. Base for diff: {self.base_ref_for_diff}")
            if not self.base_ref_for_diff: # If no specific base_ref, maybe set to current HEAD for future diffs
                # self.base_ref_for_diff = self._get_current_commit_hash() # Or handle this explicitly
                logger.info("No explicit base_ref provided. Will list all tracked files on first appropriate call, or diff against HEAD if base_ref is updated.")

    def _get_current_commit_hash(self) -> Optional[str]:
        if not self.is_git_repo:
            return None
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            logger.warning(f"Could not get current git commit hash for repo at '{self.root}'.")
            return None

    def changed_files(self, exts: Optional[List[str]] = None, first_run_behavior: str = "diff_from_base_or_head") -> List[Path]:
        """
        Determines changed files in the Git repository.

        Args:
            exts: Optional list of file extensions to filter by.
            first_run_behavior: Determines behavior on the first effective run or if no base_ref.
                "list_all_tracked": Lists all tracked files matching extensions.
                "diff_from_base_or_head": Tries to diff against base_ref_for_diff or HEAD.
                                          Falls back to list_all_tracked if diff is not possible/meaningful.
        Returns:
            A list of Path objects for changed files.
        """
        if not self.is_git_repo:
            logger.info("GitChangeTracker: Not a git repository. Returning no files based on git changes.")
            return []

        effective_exts = exts or [
            ".py", ".js", ".ts", ".java", ".go", ".sql", ".md", ".txt",
        ]
        
        cmd: List[str] = []
        use_diff = False

        current_head = self._get_current_commit_hash()

        if self.base_ref_for_diff and current_head and self.base_ref_for_diff != current_head:
            # A base_ref is set and it's different from current HEAD, so we can diff
            logger.info(f"Performing diff between {self.base_ref_for_diff} and HEAD ({current_head}).")
            cmd = ['git', 'diff', '--name-only', self.base_ref_for_diff, current_head, '--']
            cmd.extend([f'*{ext}' for ext in effective_exts]) # Add extension filters to diff
            use_diff = True
        elif first_run_behavior == "list_all_tracked" and not self.initial_run_done:
            logger.info("First run behavior: Listing all tracked files.")
            cmd = ['git', 'ls-files', '--'] + [f'*{ext}' for ext in effective_exts]
            self.initial_run_done = True # Mark that the first run (list all) has occurred
        elif first_run_behavior == "diff_from_base_or_head" and current_head and self.base_ref_for_diff == current_head:
            logger.info(f"Base ref '{self.base_ref_for_diff}' is same as current HEAD. No changes to diff. Returning empty.")
            return [] # Or potentially list untracked/modified files here if desired
        else: # Fallback or subsequent runs without a changing base_ref
            if current_head and self.base_ref_for_diff:
                logger.info(f"Defaulting to diff against provided base_ref '{self.base_ref_for_diff}' and HEAD.")
                cmd = ['git', 'diff', '--name-only', self.base_ref_for_diff, current_head, '--']
                cmd.extend([f'*{ext}' for ext in effective_exts])
                use_diff = True
            elif current_head: # No base_ref_for_diff, try diffing staged against HEAD (or list modified/untracked)
                               # For simplicity, let's list all tracked files as a fallback if no clear diff strategy
                logger.info("No specific base_ref or different HEAD for diff. Listing all tracked files as fallback.")
                cmd = ['git', 'ls-files', '--'] + [f'*{ext}' for ext in effective_exts]
            else: # Cannot get current_head
                logger.warning("Could not determine current HEAD. Listing all tracked files as a fallback.")
                cmd = ['git', 'ls-files', '--'] + [f'*{ext}' for ext in effective_exts]

        try:
            logger.debug(f"GitChangeTracker running command: {' '.join(cmd)} in {self.root}")
            output = subprocess.check_output(cmd, cwd=self.root, text=True, stderr=subprocess.PIPE)
            
            git_files_relative_paths = [p.strip() for p in output.splitlines() if p.strip()]
            
            # Convert relative paths from git output to absolute paths
            # Note: 'git diff --name-only' outputs paths relative to repo root.
            # 'git ls-files' also outputs paths relative to repo root by default.
            files_to_return = [self.root / p for p in git_files_relative_paths if (self.root / p).is_file() and (self.root / p).suffix in effective_exts]

            # If a diff was successfully performed, update base_ref_for_diff to the current_head
            # for the next incremental diff. This makes subsequent calls find changes *since this point*.
            if use_diff and current_head:
                logger.info(f"Updating base_ref_for_diff to current HEAD: {current_head}")
                self.base_ref_for_diff = current_head
                self.initial_run_done = True # A diff implies we've moved past the initial state

            return files_to_return
        except subprocess.CalledProcessError as exc:
            logger.warning(f"git command '{' '.join(exc.cmd)}' failed (code {exc.returncode}): {exc.stderr.strip()}. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in GitChangeTracker.changed_files: {e}", exc_info=True)
            return []

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

    def analyze(self, paths: List[Path]): # paths are already Path objects
        # Filter for Python files and ensure they actually exist as files
        py_files = [p for p in paths if p.suffix == ".py" and p.is_file()]
        results: Dict[str, Dict[str, int]] = {}

        if not py_files:
            logger.info("CodeAnalyzer: No Python files provided or found to analyze.")
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_path = {}
            for p in py_files:
                try:
                    # Submit a helper that reads then analyzes
                    future_to_path[pool.submit(self._read_and_analyze_single_file, p)] = p
                except Exception as e:
                    logger.error(f"Failed to submit analysis task for file {p}: {e}")
            
            for fut in as_completed(future_to_path):
                p_path = future_to_path[fut]
                try:
                    analysis_metrics = fut.result() # Result from _read_and_analyze_single_file
                    if analysis_metrics is not None: # Check if analysis was successful
                        results[str(p_path)] = analysis_metrics
                except Exception as exc:
                    # Log error for this specific file but continue with others
                    logger.error("Analysis/processing failure for file %s: %s", p_path, exc, exc_info=True)
        return results

    def _read_and_analyze_single_file(self, file_path: Path) -> Optional[Dict[str, int]]:
        """Helper method to read a file's content and then analyze it."""
        try:
            code_content = file_path.read_text(encoding="utf-8")
            return _scan(code_content) # _scan is your existing AST parsing function
        except FileNotFoundError:
            logger.warning(f"File not found during analysis: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to read or perform AST scan on file {file_path}: {e}", exc_info=True)
            return None


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
        if not self.tracker.is_git_repo:
            logger.info("Git repository not available in the current environment. "
                        "Skipping incremental codebase analysis based on git changes.")
            # Depending on requirements, you might still want to embed *all* files
            # on first run if no git, or do nothing. For now, just skipping git part.
            # If you want to embed all files if no git, that logic would go here.
            # For now, it just means no "changed" files are found via git.
            # The embedder might still be called with an empty list or a list from another source.
            return {"status": "skipped_no_git_repo", "changed_files_git": 0, "files_analyzed_metrics": 0}

        # If it IS a git repo, proceed:
        # Pass first_run_behavior='list_all_tracked' if you want to embed everything the first time
        # this AgenticCreativitySystemV2 instance is used in a git repo.
        # Or, if you manage a persistent `base_ref` across restarts, pass that.
        changed = self.tracker.changed_files(first_run_behavior="list_all_tracked")

        if not changed:
            logger.info("No changed files detected by GitChangeTracker for incremental analysis.")
            return {"status": "no_changes", "changed_files_git": 0, "files_analyzed_metrics": 0}

        logger.info(f"Found {len(changed)} files via GitChangeTracker for incremental analysis.")
        # The 'changed' list here will be based on git operations if is_git_repo was true,
        # or empty if is_git_repo was false.
        metrics = self.analyzer.analyze(changed) # analyzer now uses ThreadPoolExecutor

        files_embedded_count = 0
        for p in changed: # Only iterate if 'changed' is not empty
            try:
                self.embedder.embed_file(p)
                files_embedded_count +=1
            except Exception as exc:
                logger.warning("Embedding failed for changed file %s: %s", p, exc)
        
        analyzed_count = len(metrics)
        summary = {
            "status": "completed" if changed else "no_changes_processed",
            "changed_files_git": len(changed),
            "files_analyzed_for_metrics": analyzed_count,
            "files_embedded": files_embedded_count,
            "aggregated_loc": sum(m.get("loc", 0) for m in metrics.values()),
            "files_details": metrics
        }
        # Storing analysis results might be optional depending on runtime needs
        # For self-analysis, Nyx might want to store this.
        try:
            await self.storage.store_content(
               content_type="analysis",
               title=f"Runtime incremental analysis {datetime.utcnow().isoformat(timespec='seconds')}",
               content=json.dumps(summary, indent=2)
            )
        except Exception as e:
            logger.error(f"Failed to store content analysis: {e}")

        logger.info(f"Incremental codebase analysis completed. Git-detected changes: {len(changed)}, Analyzed for metrics: {analyzed_count}, Embedded: {files_embedded_count}.")
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
