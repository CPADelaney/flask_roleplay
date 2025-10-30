from __future__ import annotations
"""
Creative‑System core (v2.2) – **hosted semantic retrieval**
===========================================================
Implements the three requested upgrades:

1.  **Agents FileSearchTool integration** – semantic lookups route
    through the hosted vector store via the Agents stack instead of a
    local Annoy index.
2.  **Change-aware analysis** – Git change tracking feeds the
    lightweight metrics pipeline to surface modified modules.
3.  **Prompt builder** – `prepare_prompt(query, user_msg)` returns a
    context-augmented prompt ready for o3/o4-mini; it automatically
    streams the top-K retrieved snippets with delimiters.

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
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import Any, Dict, List, Optional

from rag import ask as rag_ask

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _extract_line_bounds(metadata: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    start = _coerce_int(metadata.get("line_start") or metadata.get("start_line") or metadata.get("lineStart"))
    end = _coerce_int(metadata.get("line_end") or metadata.get("end_line") or metadata.get("lineEnd"))

    range_payload = metadata.get("line_range") or metadata.get("lineRange")
    if isinstance(range_payload, dict):
        start = start or _coerce_int(range_payload.get("start") or range_payload.get("from"))
        end = end or _coerce_int(range_payload.get("end") or range_payload.get("to"))
    elif isinstance(range_payload, str):
        pieces = [piece.strip() for piece in range_payload.split("-") if piece.strip()]
        if pieces:
            start = start or _coerce_int(pieces[0])
            if len(pieces) > 1:
                end = end or _coerce_int(pieces[-1])

    return start, end


def _resolve_filepath(metadata: Dict[str, Any]) -> str:
    for key in ("filename", "filepath", "path", "file", "source", "document", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value

    memory_id = metadata.get("memory_id")
    if isinstance(memory_id, str) and memory_id.strip():
        return memory_id

    return "unknown"


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
        if changed:
            logger.debug(
                "Skipping local embedding for %s files; retrieval now routes through FileSearchTool",
                len(changed),
            )
        
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
        try:
            response = await rag_ask(
                query,
                mode="retrieval",
                limit=k,
                backend="agents",
                metadata={
                    "component": "nyx.creative.agentic_system",
                    "operation": "semantic_search",
                },
            )
        except Exception as exc:
            logger.warning("FileSearchTool retrieval failed for query '%s': %s", query, exc)
            return []

        documents: List[Dict[str, Any]] = []
        provider: Optional[str] = None

        if isinstance(response, dict):
            raw_documents = response.get("documents")
            if isinstance(raw_documents, list):
                documents = [d for d in raw_documents if isinstance(d, dict)]
            provider_value = response.get("provider")
            if isinstance(provider_value, str):
                provider = provider_value

        results: List[Dict[str, Any]] = []
        for doc in documents:
            doc_metadata = doc.get("metadata")
            if not isinstance(doc_metadata, dict):
                doc_metadata = {}

            score = _coerce_float(doc.get("score"))
            snippet_source = doc.get("text") or doc.get("content") or doc.get("value") or ""
            snippet = str(snippet_source).strip()
            filepath = _resolve_filepath(doc_metadata)
            line_start, line_end = _extract_line_bounds(doc_metadata)

            results.append(
                {
                    "score": score,
                    "snippet": snippet,
                    "filepath": filepath,
                    "line_start": line_start,
                    "line_end": line_end,
                    "metadata": doc_metadata,
                    "provider": provider or doc.get("provider"),
                }
            )

        return results

    async def prepare_prompt(self, query: str, user_msg: str, k: int = 5) -> str:
        """Return a full prompt with top‑K context snippets prepended."""
        hits = await self.semantic_search(query, k)
        ctx_parts: List[str] = []
        for index, hit in enumerate(hits):
            score = hit.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"

            filepath = hit.get("filepath") or "unknown"
            line_start = hit.get("line_start")
            line_end = hit.get("line_end")

            if line_start is not None and line_end is not None:
                location = f"{filepath} L{line_start}-{line_end}"
            elif line_start is not None:
                location = f"{filepath} L{line_start}"
            else:
                location = filepath

            snippet = str(hit.get("snippet") or "").rstrip()
            if not snippet:
                snippet = "[no content returned]"

            ctx_parts.append(
                f"# Context snippet {index + 1} (score {score_text})\n# {location}\n"
                + indent(snippet, "# ")
            )
        context = "\n\n".join(ctx_parts)
        return f"{context}\n\n# User question\n{user_msg}"

    async def write_story(self, prompt: str, length: str = "medium", genre: str = None) -> Dict[str, Any]:
        """Generate a creative story based on the provided prompt."""
        # Use the semantic search and prompt builder to find relevant context
        context_enriched_prompt = await self.prepare_prompt(prompt, f"Write a {length} story about {prompt}" + (f" in {genre} genre" if genre else ""))
        
        # Store the generated content
        result = await self.storage.store_content(
            content_type="story",
            title=f"Story: {prompt[:50]}",
            content=f"# {prompt}\n\n[Story would be generated here with a more advanced generation system]"
        )
        
        return result
    
    async def write_poem(self, topic: str, style: str = None) -> Dict[str, Any]:
        """Generate a poem on the specified topic and in the specified style."""
        style_prompt = f" in {style} style" if style else ""
        context_enriched_prompt = await self.prepare_prompt(topic, f"Write a poem about {topic}{style_prompt}")
        
        result = await self.storage.store_content(
            content_type="poem",
            title=f"Poem: {topic[:50]}",
            content=f"# {topic}\n\n[Poem would be generated here with a more advanced generation system]"
        )
        
        return result
    
    async def write_lyrics(self, theme: str, genre: str = "pop") -> Dict[str, Any]:
        """Generate song lyrics based on theme and genre."""
        context_enriched_prompt = await self.prepare_prompt(theme, f"Write {genre} song lyrics about {theme}")
        
        result = await self.storage.store_content(
            content_type="lyrics",
            title=f"{genre.capitalize()} Song: {theme[:50]}",
            content=f"# {theme} ({genre})\n\n[Lyrics would be generated here with a more advanced generation system]"
        )
        
        return result
    
    async def write_journal(self, topic: str, perspective: str = "first-person") -> Dict[str, Any]:
        """Generate a journal entry on the specified topic."""
        context_enriched_prompt = await self.prepare_prompt(topic, f"Write a {perspective} journal entry about {topic}")
        
        result = await self.storage.store_content(
            content_type="journal",
            title=f"Journal: {topic[:50]}",
            content=f"# {topic}\n\n[Journal entry would be generated here with a more advanced generation system]"
        )
        
        return result
    
    async def write_and_execute_code(self, task: str, language: str = "python") -> Dict[str, Any]:
        """Generate and execute code to solve a specific task."""
        context_enriched_prompt = await self.prepare_prompt(task, f"Write {language} code to {task}")
        
        # Placeholder for code generation
        code = f"# {language} code to {task}\n\n# [Code would be generated and executed here]"
        
        result = await self.storage.store_content(
            content_type="code",
            title=f"Code: {task[:50]}",
            content=code,
            metadata={"language": language, "task": task}
        )
        
        return result
    
    # Connect to the CodeAnalyzer for these methods
    async def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze a Python module structure."""
        # This would ideally be connected to CodeAnalyzer.analyze_module
        from nyx.creative.analysis_sandbox import CodeAnalyzer
        analyzer = CodeAnalyzer(self.storage)
        return await analyzer.analyze_module(module_path)
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Review code for improvements and issues."""
        # This would ideally be connected to CodeAnalyzer.review_code
        from nyx.creative.analysis_sandbox import CodeAnalyzer
        analyzer = CodeAnalyzer(self.storage)
        return await analyzer.review_code(code, language)
    
    # Connect to the CapabilityAssessmentSystem for this method
    async def assess_capabilities(self, goal: str) -> Dict[str, Any]:
        """Assess capabilities required for a specific goal."""
        # This would ideally be connected to CapabilityAssessmentSystem.assess_required_capabilities
        from nyx.creative.capability_system import CapabilityAssessmentSystem
        assessor = CapabilityAssessmentSystem(creative_content_system=self.storage)
        return await assessor.assess_required_capabilities(goal)

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
