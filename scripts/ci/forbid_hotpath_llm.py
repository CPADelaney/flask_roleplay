#!/usr/bin/env python3
"""
CI Guard: Forbid direct Runner.run invocations outside Nyx gateway/core/tests code.

The LLM hot-path policy now requires **all** synchronous entrypoints to invoke the
central `nyx.gateway.llm_gateway.execute(...)` wrapper instead of touching the
Runner directly. This script scans every Python file in the repository and fails
the build when it finds a call to `Runner.run` in any path that is not part of the
Nyx gateway, Nyx core, or tests packages. The goal is to prevent accidental direct
Runner usage in request handlers, orchestrators, or background tasks that should
route through the gateway.

Usage:
    python scripts/ci/forbid_hotpath_llm.py

Exit codes:
    0: No violations found
    1: Violations found
    2: Script error
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Regex pattern to detect direct Runner usage
RUNNER_PATTERN = (r"\bRunner\.run\s*\(", "Runner.run call")

# Allowed path prefixes (relative to repo root) where Runner.run() is legitimate
ALLOWED_PREFIXES = (
    Path("nyx/gateway"),
    Path("nyx/core"),
    Path("tests"),
)


def is_allowed_path(rel_path: Path) -> bool:
    """Return True if the file is inside an allowed prefix."""
    return any(rel_path.parts[:len(prefix.parts)] == prefix.parts for prefix in ALLOWED_PREFIXES)


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for forbidden Runner.run() usages.

    Returns:
        List of (line_number, pattern_name, line_content) tuples
    """
    violations: List[Tuple[int, str, str]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            pattern_re, pattern_name = RUNNER_PATTERN
            if re.search(pattern_re, line):
                violations.append((line_num, pattern_name, line.strip()))

    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)

    return violations


def main() -> int:
    """Main entry point."""
    repo_root = Path.cwd()
    print(f"🔍 Scanning repository for direct Runner.run calls in {repo_root}")
    print(f"   Allowed prefixes: {[str(prefix) for prefix in ALLOWED_PREFIXES]}")
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
