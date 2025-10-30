#!/usr/bin/env python3
"""Fail the build if legacy embedding helpers are referenced outside allowlists."""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PatternCheck:
    """A regex check bound to a descriptive name."""

    name: str
    regex: re.Pattern[str]
    description: str


CHECKS: List[PatternCheck] = [
    PatternCheck(
        name="HUGGINGFACE_EMBEDDINGS",
        regex=re.compile(r"\bHuggingFaceEmbeddings\b"),
        description=(
            "Legacy HuggingFaceEmbeddings usage detected."
            " Switch to the hosted Agents vector store APIs."
        ),
    ),
    PatternCheck(
        name="OPENAI_EMBEDDINGS",
        regex=re.compile(r"\bOpenAIEmbeddings\b"),
        description=(
            "Legacy OpenAIEmbeddings usage detected."
            " Switch to the hosted Agents vector store APIs."
        ),
    ),
]


def load_allowlist(path: Path) -> Dict[str, List[str]]:
    """Parse an allowlist file into {pattern_name: [glob, ...]} mappings."""

    allowlist: Dict[str, List[str]] = {check.name: [] for check in CHECKS}
    if not path.exists():
        return allowlist

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Support inline comments using `#` by stripping everything after it.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = [segment.strip() for segment in line.split("|", 1)]
        if len(parts) != 2:
            raise ValueError(
                f"Malformed allowlist entry at {path}:{lineno}: expected 'PATTERN|glob', got {raw_line!r}"
            )

        pattern_name, glob = parts
        if pattern_name not in allowlist:
            known = ", ".join(sorted(allowlist))
            raise ValueError(
                f"Unknown pattern '{pattern_name}' at {path}:{lineno}."
                f" Known patterns: {known}"
            )
        allowlist[pattern_name].append(glob)

    return allowlist


def list_repo_files(root: Path) -> List[Path]:
    """Return tracked files under *root* using git ls-files."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [root / Path(line) for line in result.stdout.splitlines() if line]


def is_allowlisted(check: PatternCheck, relative_path: str, allowlist: Dict[str, List[str]]) -> bool:
    """Return True if *relative_path* is allowlisted for *check*."""

    patterns = allowlist.get(check.name, [])
    return any(fnmatch.fnmatch(relative_path, glob) for glob in patterns)


def scan_file(path: Path, check: PatternCheck) -> List[tuple[int, str]]:
    """Return a list of (line_number, line_content) that violate *check*."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    violations: List[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if check.regex.search(line):
            violations.append((lineno, line.rstrip()))
    return violations


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).with_name("forbid_legacy_embeddings_allowlist.txt"),
        help="Path to the allowlist file containing PATTERN|glob entries.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    allowlist = load_allowlist(args.allowlist)

    violations_found = False
    repo_files = list_repo_files(repo_root)

    for file_path in repo_files:
        if file_path.suffix != ".py":
            continue
        relative = file_path.relative_to(repo_root).as_posix()
        for check in CHECKS:
            if is_allowlisted(check, relative, allowlist):
                continue

            matches = scan_file(file_path, check)
            if not matches:
                continue

            violations_found = True
            for lineno, line in matches:
                print(
                    f"{relative}:{lineno}: {check.name}: {check.description}\n"
                    f"    {line}"
                )

    if violations_found:
        print("\nLegacy embedding references detected. Update the implementation to use the"
              " Agents vector store APIs or extend the allowlist if this is intentional.")
        return 1

    print("No legacy embedding patterns found. ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
