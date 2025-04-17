# nyx/core/repo_refactor_pipeline.py

from __future__ import annotations
"""
Repo‑Refactor Pipeline (v1.2)
=============================
GitHub‑friendly, non‑interactive pipeline:

* Uses OpenAI **chat.responses** endpoint.
* Stores LLM patch as `_llm_patch.md` and in SQLite.
* **If not auto‑approved**, pushes a *draft PR* so you can review in the
  GitHub UI.
* If an approval callback (or later, a PR comment trigger) returns `True`,
  applies patch → runs `pytest` → commits to a new branch and opens a
  normal PR.

No TTY interaction required.
"""

import asyncio, json, os, subprocess, textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from logic.chatgpt_integration import get_openai_client
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    AsyncOpenAI,
    OpenAIResponsesModel
)

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
# Static analysis helper
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
            r = subprocess.run([
                "ruff", "check", "--format", "json", *rels
            ], cwd=self.repo_root, text=True, capture_output=True, check=False)
            if r.stdout:
                for item in json.loads(r.stdout):
                    issues.append(Issue(item["filename"], item["location"]["row"], "ruff", item["code"], item["message"]))
        except FileNotFoundError:
            pass

        # mypy
        try:
            subprocess.run([
                "mypy", "--show-error-codes", "--no-color-output", "--no-error-summary", "--json-report", "_mypy_report", *rels
            ], cwd=self.repo_root, text=True, capture_output=True, check=False)
            report = self.repo_root / "_mypy_report" / "index.json"
            if report.exists():
                data = json.loads(report.read_text())
                for err in data.get("errors", []):
                    issues.append(Issue(err["path"], err["line"], "mypy", err.get("code", "typing"), err["message"]))
        except FileNotFoundError:
            pass

        # bandit
        try:
            r = subprocess.run([
                "bandit", "-q", "-f", "json", "-r", *rels
            ], cwd=self.repo_root, text=True, capture_output=True, check=False)
            if r.stdout:
                data = json.loads(r.stdout)
                for res in data.get("results", []):
                    issues.append(Issue(res["filename"], res["line_number"], "bandit", res["test_id"], res["issue_text"]))
        except FileNotFoundError:
            pass

        return issues

# ---------------------------------------------------------------------------
# OpenAI chat.responses helper
# ---------------------------------------------------------------------------


async def call_llm(prompt: str, model: str = "gpt-4o"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "<!-- OPENAI_API_KEY not set – returning prompt for debug -->\n" + prompt[:800]
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Create the base parameters
    params = {
        "model": model,
        "input": prompt,
        "instructions": "You are an autonomous repo steward AI that suggests minimal, high‑impact patches."
    }
    
    # Only add temperature for models that support it
    # O-series models (like o4-mini) don't support temperature
    if not model.startswith("o"):
        params["temperature"] = 0.15
        params["top_p"] = 0.9

    # Add before the call_llm line
    print(f"Prompt length: {len(prompt)}")
    print(f"Lint issues found: {len(issue)}")
    # For debugging, you could temporarily save the prompt
    Path("debug_prompt.txt").write_text(prompt, encoding="utf-8")
    
    # Make the API call
    response = await client.responses.create(**params)
    
    # Process the response
    result = ""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    result += content.text
    
    return result
  


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class RepoRefactorPipeline:
    """Run the refactor loop with optional human approval."""

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.system = AgenticCreativitySystem(repo_root)
        self.static = StaticAnalyzer(repo_root)

    async def run(
        self,
        goal: str,
        *,
        approve_callback: Optional[Callable[[str], bool]] = None,
        open_pr: bool = False,
        model: str = "o4-mini",
    ) -> Dict[str, Any]:
        # Step 1: ingest / embed changed files
        full_scan = os.getenv("FULL_SCAN", "false").lower() == "true"
        changed = (list(self.system.tracker.root.rglob("*"))
                   if full_scan
                   else self.system.tracker.changed_files())
        await self.system.incremental_codebase_analysis()

        # Step 2: static analysis on changed paths
        issues = await self.static.run(changed)
        issues_md = "\n".join(
            f"- `{i.tool}` {i.path}:{i.line} `{i.code}` {i.msg}" for i in issues[:200]
        )

        # Step 3: build prompt + call model
        base_msg = (
            f"You are the repo steward. Goal: {goal}. "
            "Below are lint issues and context snippets. Suggest concrete patches in unified diff format."
        )
        prompt = await self.system.prepare_prompt(goal, base_msg + "\n\n## Lint issues\n" + issues_md, k=6)
        response = await call_llm(prompt, model=model)

        # Step 4: save suggestion
        patch_path = self.repo_root / "_llm_patch.md"
        patch_path.write_text(response, encoding="utf-8")
        await self.system.storage.store_content(
            content_type="assessment",
            title=f"Refactor suggestion {datetime.utcnow().isoformat(timespec='seconds')}",
            content=response,
            metadata={"goal": goal, "issues": len(issues)},
        )

        # Step 5: approval gate
        approved = approve_callback(response) if approve_callback else False
        if not approved:
            if open_pr:
                await self._open_draft_pr(goal, patch_path)
            return {"issues": len(issues), "chars": len(response), "applied": False, "pr_opened": bool(open_pr)}

        # Step 6: apply patch & run tests
        try:
            subprocess.run(["git", "apply", str(patch_path)], cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            return {"error": f"git apply failed: {exc}", "applied": False}

        tests_ok = subprocess.run(["pytest", "-q"], cwd=self.repo_root).returncode == 0
        if not tests_ok:
            subprocess.run(["git", "apply", "-R", str(patch_path)], cwd=self.repo_root)
            return {"issues": len(issues), "chars": len(response), "applied": False, "tests": "failed"}

        branch = f"auto/refactor-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        subprocess.run(["git", "checkout", "-b", branch], cwd=self.repo_root)
        subprocess.run(["git", "add", "-u"], cwd=self.repo_root)
        subprocess.run(["git", "commit", "-m", goal], cwd=self.repo_root)
        if open_pr:
            subprocess.run(["git", "push", "-u", "origin", branch], cwd=self.repo_root)
            subprocess.run(["gh", "pr", "create", "--fill", "--head", branch], cwd=self.repo_root)

        return {"issues": len(issues), "chars": len(response), "applied": True, "tests": "passed"}

    # ---------------------------------------------------------
    async def _open_draft_pr(self, goal: str, patch_file: Path):
        branch = f"auto/refactor-suggestion-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        subprocess.run(["git", "checkout", "-b", branch], cwd=self.repo_root)
        subprocess.run(["git", "add", str(patch_file)], cwd=self.repo_root)
        subprocess.run(["git", "commit", "-m", f'draft: {goal} (LLM suggestion)'], cwd=self.repo_root)
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=self.repo_root)
        subprocess.run(["gh", "pr", "create", "--fill", "--head", branch, "--draft"], cwd=self.repo_root)

# ---------------------------------------------------------------------------
# CLI helper (non‑interactive)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("goal", help="High‑level refactor goal, e.g. 'improve db layer'")
    parser.add_argument("--pr", action="store_true", help="Open GitHub draft PR with suggestion")
    args = parser.parse_args()

    asyncio.run(RepoRefactorPipeline().run(args.goal, open_pr=args.pr))
