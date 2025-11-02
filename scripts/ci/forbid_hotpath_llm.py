#!/usr/bin/env python3
"""
CI Guard: Forbid direct Runner.run invocations outside Nyx gateway/worker/test code.

The LLM hot-path policy now requires **all** synchronous entrypoints to invoke the
central `nyx.gateway.llm_gateway.execute(...)` wrapper instead of touching the
Runner directly. This script scans every Python file in the repository and fails
the build when it finds a call to `Runner.run` in any path that is not part of the
Nyx gateway, worker packages, Nyx core, or tests. The goal is to prevent accidental
direct Runner usage in request handlers, orchestrators, or background tasks that
should route through the gateway or dedicated worker modules.

Usage:
    python scripts/ci/forbid_hotpath_llm.py

Exit codes:
    0: No violations found
    1: Violations found
    2: Script error
"""

import io
import os
import sys
import tokenize
from pathlib import Path
from typing import List, Tuple

# Description used when reporting Runner.run violations
RUNNER_PATTERN_NAME = "Runner.run call"

# Allowed path prefixes (relative to repo root) where Runner.run() is legitimate
ALLOWED_PREFIXES = (
    Path("nyx/gateway"),
    Path("nyx/core"),
    Path("tests"),
)

WORKER_SEGMENT = "workers"


def is_allowed_path(rel_path: Path) -> bool:
    """Return True if the file is inside an allowed prefix."""

    if any(rel_path.parts[:len(prefix.parts)] == prefix.parts for prefix in ALLOWED_PREFIXES):
        return True

    if rel_path.parts and rel_path.parts[0] == "nyx" and WORKER_SEGMENT in rel_path.parts:
        return True

    return False


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for forbidden Runner.run() usages.

    The scanner tokenizes the file so that string literals and comments are
    ignored. Only actual code tokens (NAME/OP) are considered when detecting the
    `Runner.run(` call pattern.

    Returns:
        List of (line_number, pattern_name, line_content) tuples
    """
    violations: List[Tuple[int, str, str]] = []
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return violations

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.splitlines()

    pattern_name = RUNNER_PATTERN_NAME

    index = 0
    while index < len(tokens):
        tok = tokens[index]
        if tok.type == tokenize.NAME and tok.string == "Runner":
            # Look ahead for `.run` followed by an opening parenthesis, ignoring
            # non-code tokens (newlines/indent/dedent) between pieces.
            lookahead = index + 1
            if lookahead < len(tokens) and tokens[lookahead].type == tokenize.OP and tokens[lookahead].string == ".":
                lookahead += 1
                if lookahead < len(tokens) and tokens[lookahead].type == tokenize.NAME and tokens[lookahead].string == "run":
                    # Find the next meaningful token after `run`.
                    lookahead += 1
                    while lookahead < len(tokens) and tokens[lookahead].type in {
                        tokenize.NL,
                        tokenize.NEWLINE,
                        tokenize.INDENT,
                        tokenize.DEDENT,
                        tokenize.COMMENT,
                    }:
                        lookahead += 1
                    if lookahead < len(tokens) and tokens[lookahead].type == tokenize.OP and tokens[lookahead].string == "(":
                        line_num = tok.start[0]
                        # Provide the full line for context in the violation report.
                        line_content = lines[line_num - 1].strip() if 0 <= line_num - 1 < len(lines) else ""
                        violations.append((line_num, pattern_name, line_content))
                        index = lookahead  # Skip ahead to avoid duplicate reports on same call
                        continue
        index += 1

    return violations


def main() -> int:
    """Main entry point."""
    repo_root = Path.cwd()
    print(f"🔍 Scanning repository for direct Runner.run calls in {repo_root}")
    allowed_prefixes = [str(prefix) for prefix in ALLOWED_PREFIXES]
    print(f"   Allowed prefixes: {allowed_prefixes} + nyx/**/{WORKER_SEGMENT}/*")
    print()

    all_violations = {}
    file_count = 0

    # Find all Python files
    for py_file in repo_root.rglob("*.py"):
        # Skip virtual environments, caches, etc.
        if any(part.startswith(".") or part in ["venv", "env", "__pycache__", "node_modules"]
               for part in py_file.parts):
            continue

        rel_path = py_file.relative_to(repo_root)

        if is_allowed_path(rel_path):
            continue

        file_count += 1
        violations = scan_file(py_file)

        if violations:
            all_violations[rel_path] = violations

    # Report results
    if not all_violations:
        print(f"✅ SUCCESS: No forbidden Runner.run calls found in {file_count} scanned files")
        print()
        print("All non-gateway/core/tests code is clean! 🎉")
        return 0

    # Print violations
    print(f"❌ FAILURE: Found direct Runner.run calls in {len(all_violations)} files outside the gateway/core/tests allowlist:\n")

    for file_path, violations in sorted(all_violations.items()):
        print(f"📁 {file_path}")
        for line_num, pattern_name, line_content in violations:
            print(f"   Line {line_num}: {pattern_name}")
            print(f"      {line_content}")
        print()

    print(f"\n{'='*80}")
    print(f"❌ TOTAL VIOLATIONS: {sum(len(v) for v in all_violations.values())} "
          f"in {len(all_violations)} files")
    print(f"{'='*80}\n")

    print("🔧 How to fix:")
    print("   • Move the call into nyx/gateway/ or nyx/core/, or expose it via a gateway helper for callers")
    print("   • Otherwise replace direct Runner usage with nyx.gateway.llm_gateway.execute(...)")
    print()

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
