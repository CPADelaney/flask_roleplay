# nyx/core/repo_refactor_pipeline.py

from __future__ import annotations
"""
Repo‑Refactor Pipeline (v1.1)
=============================
Same orchestrator as v1, but the **LLM call now uses OpenAI’s newer
`chat.responses` endpoint** (released 2025‑Q1) instead of the older
`chat.completions` API.

No other behaviour changes.
"""

import asyncio, json, os, subprocess, tempfile, textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import openai
from nyx.creative.agentic_system import AgenticCreativitySystem  # shim to v2.2

# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    path: str
    line: int
    tool: str  # ruff | mypy | bandit
    code: str
    msg: str

# ---------------------------------------------------------------------------
# Static analysis helper (unchanged)
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    async def run(self, files: List[Path]) -> List[Issue]:
        issues: List[Issue] = []
        if not files:
            return issues
        rels = [str(p.relative_to(self.repo_root)) for p in files]

        # ruff
        try:
            r = subprocess.run(["ruff", "check", "--format", "json"] + rels, cwd=self.repo_root, text=True, capture_output=True, check=False)
            if r.stdout:
                for item in json.loads(r.stdout):
                    issues.append(Issue(item["filename"], item["location"]["row"], "ruff", item["code"], item["message"]))
        except FileNotFoundError:
            pass

        # mypy
        try:
            subprocess.run(["mypy", "--show-error-codes", "--no-color-output", "--no-error-summary", "--json-report", "_mypy_report"] + rels, cwd=self.repo_root, text=True, capture_output=True, check=False)
            report_path = self.repo_root / "_mypy_report" / "index.json"
            if report_path.exists():
                data = json.loads(report_path.read_text())
                for err in data.get("errors", []):
                    issues.append(Issue(err["path"], err["line"], "mypy", err.get("code", "typing"), err["message"]))
        except FileNotFoundError:
            pass

        # bandit
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
# OpenAI chat.responses wrapper
# ---------------------------------------------------------------------------

async def call_llm(prompt: str, model: str = "o3-turbo") -> str:
    """Call the new OpenAI Chat Responses endpoint (async)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "<!-- OPENAI_API_KEY not set – returning prompt for debug -->\n" + prompt[:1200]

    openai.api_key = api_key
    try:
        resp = await openai.ChatResponses.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "You are an autonomous repo steward AI that suggests minimal, high‑impact patches."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"<!-- LLM call failed: {exc} -->\n" + prompt[:1200]

# ---------------------------------------------------------------------------
# Pipeline orchestrator (minor update: passes model response through)
# ---------------------------------------------------------------------------

class RepoRefactorPipeline:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.system = AgenticCreativitySystem(repo_root)
        self.static = StaticAnalyzer(repo_root)

    async def run(self, goal: str, open_pr: bool = False):
        changed = self.system.tracker.changed_files()
        await self.system.incremental_codebase_analysis()

        issues = await self.static.run(changed)
        issues_md = "\n".join(f"- `{i.tool}` {i.path}:{i.line} `{i.code}` {i.msg}" for i in issues[:200])

        base_msg = f"You are the repo steward. Goal: {goal}. Below are lint issues and context snippets. Suggest concrete patches."
        prompt = await self.system.prepare_prompt(goal, base_msg + "\n\n## Lint issues\n" + issues_md, k=6)
        response = await call_llm(prompt)

        await self.system.storage.store_content(
            content_type="assessment",
            title=f"Refactor suggestion {datetime.utcnow().isoformat(timespec='seconds')}",
            content=response,
            metadata={"goal": goal, "issues": len(issues)},
        )

        if open_pr:
            branch = f"auto/refactor-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            subprocess.run(["git", "checkout", "-b", branch], cwd=self.repo_root)
            patch_file = self.repo_root / "_llm_patch.md"
            patch_file.write_text(response, encoding="utf-8")
            subprocess.run(["git", "add", str(patch_file)], cwd=self.repo_root)
            subprocess.run(["git", "commit", "-m", "LLM refactor suggestion"], cwd=self.repo_root)
            subprocess.run(["git", "push", "-u", "origin", branch], cwd=self.repo_root)
            subprocess.run(["gh", "pr", "create", "--fill", "--head", branch], cwd=self.repo_root)

        return {"issues": len(issues), "chars": len(response)}

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("goal", help="high‑level refactor goal, e.g. 'improve db layer'")
    parser.add_argument("--pr", action="store_true", help="open a GitHub PR with the patch")
    args = parser.parse_args()

    asyncio.run(RepoRefactorPipeline().run(args.goal, open_pr=args.pr))
