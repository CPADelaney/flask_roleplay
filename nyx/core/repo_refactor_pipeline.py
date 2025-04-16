# nyx/core/repo_refactor_pipeline.py

from __future__ import annotations
"""
Repo‑Refactor Pipeline (v1)
===========================
A standalone orchestrator that plugs into the existing Creative‑System
(v2.2) **without changing any imports in the rest of your repo.**

Features
--------
* Runs `incremental_codebase_analysis()` to refresh the vector index.
* Executes *static linters* (`ruff`, `mypy`, `bandit`) only on changed
  files; parses their JSON output into a consolidated **IssueList**.
* Creates a *refactor prompt* using `prepare_prompt()` and feeds the
  context to an LLM (placeholder hook).  The LLM's response is stored via
  `CreativeContentSystem` for auditability.
* Optionally writes the LLM‑generated patch to a temporary branch and
  opens a pull‑request (GitHub CLI required).

Public entry point: `async run_pipeline(goal: str, open_pr: bool = False)`

No other modules need to change – this file just imports
`AgenticCreativitySystem` from the existing namespace shim.
"""

import asyncio, json, os, subprocess, tempfile, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from nyx.creative.agentic_system import AgenticCreativitySystem  # shim to v2.2

# ---------------------------------------------------------------------------
# Helper dataclass for static‑analysis issues
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    path: str
    line: int
    tool: str  # ruff | mypy | bandit
    code: str
    msg: str

# ---------------------------------------------------------------------------
# Static‑analysis runner
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    """Runs linters only on the provided paths and returns Issue records."""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    async def run(self, files: List[Path]) -> List[Issue]:
        issues: List[Issue] = []
        if not files:
            return issues
        rels = [str(p.relative_to(self.repo_root)) for p in files]

        # --- ruff ---
        try:
            r = subprocess.run(["ruff", "check", "--format", "json"] + rels, cwd=self.repo_root, text=True, capture_output=True, check=False)
            if r.stdout:
                for item in json.loads(r.stdout):
                    issues.append(Issue(item["filename"], item["location"]["row"], "ruff", item["code"], item["message"]))
        except FileNotFoundError:
            pass

        # --- mypy ---
        try:
            r = subprocess.run(["mypy", "--show-error-codes", "--no-color-output", "--no-error-summary", "--json-report", "_mypy_report"] + rels, cwd=self.repo_root, text=True, capture_output=True, check=False)
            report_path = self.repo_root / "_mypy_report" / "index.json"
            if report_path.exists():
                data = json.loads(report_path.read_text())
                for err in data.get("errors", []):
                    issues.append(Issue(err["path"], err["line"], "mypy", err.get("code", "typing"), err["message"]))
        except FileNotFoundError:
            pass

        # --- bandit ---
        try:
            r = subprocess.run(["bandit", "-q", "-f", "json", "-r"] + rels, cwd=self.repo_root, text=True, capture_output=True, check=False)
            if r.stdout:
                data = json.loads(r.stdout)
                for res in data.get("results", []):
                    issues.append(Issue(res["filename"], res["line_number"], "bandit", res["test_id"], res["issue_text"]))
        except FileNotFoundError:
            pass

        return issues

# ---------------------------------------------------------------------------
# LLM call placeholder (swap with your preferred model)
# ---------------------------------------------------------------------------

async def call_llm(prompt: str) -> str:
    """Placeholder – replace with o3/o4‑mini call."""
    return "# TODO: model integration\n" + textwrap.indent(prompt[:1000], "# ")

# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class RepoRefactorPipeline:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.system = AgenticCreativitySystem(repo_root)
        self.static = StaticAnalyzer(repo_root)

    async def run(self, goal: str, open_pr: bool = False):
        changed = self.system.tracker.changed_files()
        await self.system.incremental_codebase_analysis()

        # ----- static analysis on changed files -----
        issues = await self.static.run(changed)
        issues_md = "\n".join(f"- `{i.tool}` {i.path}:{i.line} `{i.code}` {i.msg}" for i in issues[:200])

        # ----- build refactor prompt -----
        base_msg = f"You are the repo steward. Goal: {goal}. Below are lint issues and context snippets. Suggest concrete patches."
        prompt = await self.system.prepare_prompt(goal, base_msg + "\n\n## Lint issues\n" + issues_md, k=6)
        llm_response = await call_llm(prompt)

        # ----- store the suggestion -----
        await self.system.storage.store_content(
            content_type="assessment",
            title=f"Refactor suggestion {datetime.utcnow().isoformat(timespec='seconds')}",
            content=llm_response,
            metadata={"goal": goal, "issues": len(issues)},
        )

        # ----- optional PR -----
        if open_pr:
            branch = f"auto/refactor-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            subprocess.run(["git", "checkout", "-b", branch], cwd=self.repo_root)
            patch_path = self.repo_root / "_llm_patch.md"
            patch_path.write_text(llm_response)
            subprocess.run(["git", "add", str(patch_path)], cwd=self.repo_root)
            subprocess.run(["git", "commit", "-m", "LLM refactor suggestion"], cwd=self.repo_root)
            subprocess.run(["git", "push", "-u", "origin", branch], cwd=self.repo_root)
            subprocess.run(["gh", "pr", "create", "--fill", "--head", branch], cwd=self.repo_root)

        return {"issues": len(issues), "llm_chars": len(llm_response)}

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("goal", help="high‑level refactor goal, e.g. 'improve db layer'")
    parser.add_argument("--pr", action="store_true", help="open a GitHub PR with the patch")
    args = parser.parse_args()

    asyncio.run(RepoRefactorPipeline().run(args.goal, open_pr=args.pr))
