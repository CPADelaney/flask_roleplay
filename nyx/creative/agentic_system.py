# nyx/creative/agentic_system.py

from __future__ import annotations
"""
Refactored Creative‑System core (v2)
====================================
This module replaces the legacy *agentic_system* stack **without breaking
existing import paths**.  Key points:

* **SQLiteContentSystem** – persists metadata in *ai_creations/content.db*
* **GitChangeTracker** – only re‑analyses files `git diff` reports as changed
* **ParallelCodeAnalyzer** – multi‑process AST metrics
* **AgenticCreativitySystemV2** – new orchestrator
* **Compatibility shims** – aliases so old code that imports
  `AgenticCreativitySystem`, `CreativeContentSystem`, or `ContentType` keeps
  working.

Drop the file at *nyx/creative/agentic_system.py* and remove the old
`AgenticCreativitySystem` implementation.
"""

import ast
import enum
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Git‑aware change tracker
# ---------------------------------------------------------------------------

class GitChangeTracker:
    """Returns paths changed since *base_ref* (defaults to HEAD)."""

    def __init__(self, repo_root: str = ".", base_ref: str = "HEAD") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.base_ref = base_ref

    def changed_files(self, exts: Optional[List[str]] = None) -> List[Path]:
        exts = exts or [".py"]
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--name-only", self.base_ref],
                cwd=self.repo_root,
                text=True,
            )
            files = [self.repo_root / p.strip() for p in diff.splitlines() if p.strip()]
        except subprocess.CalledProcessError as exc:
            logger.warning("git diff failed (%s); scanning entire repo", exc)
            files = list(self.repo_root.rglob("*"))
        return [p for p in files if p.suffix in exts and p.exists()]

# ---------------------------------------------------------------------------
# 2. SQLite‑backed content repository
# ---------------------------------------------------------------------------

class SQLiteContentSystem:
    """Drop‑in replacement for the old CreativeContentSystem using SQLite."""

    def __init__(self, db_path: str = "ai_creations/content.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._bootstrap()

    def _bootstrap(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS content (
                id         TEXT PRIMARY KEY,
                type       TEXT NOT NULL,
                title      TEXT,
                filepath   TEXT,
                created_at TEXT,
                metadata   TEXT
            )"""
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_type_created ON content(type, created_at DESC)"
        )
        self.conn.commit()

    # -------------------------------------------------------
    # Public API – mirrors old CreativeContentSystem subset
    # -------------------------------------------------------

    async def store_content(
        self,
        *,
        content_type: str,
        title: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        cid = hashlib.sha1(f"{title}{datetime.utcnow()}".encode()).hexdigest()
        dir_path = Path("ai_creations") / content_type
        dir_path.mkdir(parents=True, exist_ok=True)
        ext = "md" if content_type in {"analysis", "assessment"} else "txt"
        fp = dir_path / f"{cid}.{ext}"
        fp.write_text(content, encoding="utf‑8")

        self.conn.execute(
            "INSERT INTO content VALUES (?,?,?,?,?,?)",
            (
                cid,
                content_type,
                title,
                str(fp),
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()
        return {"id": cid, "filepath": str(fp)}

    async def list_content(self, content_type: str | None = None, *, limit: int = 100, offset: int = 0):
        cur = self.conn.cursor()
        if content_type:
            cur.execute(
                "SELECT id,title,filepath,created_at FROM content WHERE type=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (content_type, limit, offset),
            )
        else:
            cur.execute(
                "SELECT id,title,filepath,created_at FROM content ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [
            {"id": r[0], "title": r[1], "filepath": r[2], "created_at": r[3]} for r in cur.fetchall()
        ]

# ---------------------------------------------------------------------------
# 3. Parallel AST metrics
# ---------------------------------------------------------------------------

def _scan(code: str) -> Dict[str, int]:
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
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers or os.cpu_count() or 4

    def analyze(self, paths: List[Path]) -> Dict[str, Dict[str, int]]:
        results: Dict[str, Dict[str, int]] = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(p.read_text, encoding="utf‑8"): p for p in paths}
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    results[str(path)] = _scan(fut.result())
                except Exception as exc:
                    logger.error("AST failure %s: %s", path, exc)
        return results

# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

class AgenticCreativitySystemV2:
    """Core steward used by Nyx; supersedes the old AgenticCreativitySystem."""

    def __init__(self, repo_root: str = ".", review_interval_days: int = 7):
        self.tracker = GitChangeTracker(repo_root)
        self.storage = SQLiteContentSystem()
        self.analyzer = ParallelCodeAnalyzer()
        self.review_interval_days = review_interval_days

    # ---- public API (minimal)

    async def incremental_codebase_analysis(self) -> Dict[str, Any]:
        changed = self.tracker.changed_files()
        metrics = self.analyzer.analyze(changed)
        summary = {
            "changed_files": len(changed),
            "aggregated_loc": sum(m["loc"] for m in metrics.values()),
            "files": metrics,
        }
        await self.storage.store_content(
            content_type="analysis",
            title=f"Incremental analysis {datetime.utcnow().isoformat(timespec='seconds')}",
            content=json.dumps(summary, indent=2),
        )
        return summary

# ---------------------------------------------------------------------------
# 5. Compatibility layer – keep old imports working
# ---------------------------------------------------------------------------

class _ContentType(enum.Enum):
    STORY = "story"
    POEM = "poem"
    LYRICS = "lyrics"
    JOURNAL = "journal"
    CODE = "code"
    ANALYSIS = "analysis"
    ASSESSMENT = "assessment"

# Aliases so existing `from creative.agentic_system import …` lines succeed
ContentType = _ContentType  # type: ignore
CreativeContentSystem = SQLiteContentSystem  # type: ignore
AgenticCreativitySystem = AgenticCreativitySystemV2  # type: ignore

# ---------------------------------------------------------------------------
# 6. Demo entry‑point (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:  # pragma: no cover
        system = AgenticCreativitySystemV2()
        rpt = await system.incremental_codebase_analysis()
        print(json.dumps(rpt, indent=2)[:800] + "…")

    asyncio.run(_demo())
