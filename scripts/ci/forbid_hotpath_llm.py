#!/usr/bin/env python3
"""
CI Guard: Forbid blocking LLM calls in hot-path code.

This script scans hot-path modules (event handlers, game loop, synchronous APIs)
for blocking LLM patterns like Runner.run(), llm_json(), make_autonomous_decision(),
etc. It fails the build if such patterns are found outside allowed locations
(tasks/, scripts/, migrations/, tests/).

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
from typing import List, Tuple, Set

# Blocking patterns to detect (regex patterns)
BLOCKING_PATTERNS = [
    (r"\bRunner\.run\s*\(", "Runner.run()"),
    (r"\bllm_json\s*\(", "llm_json()"),
    (r"\bmake_autonomous_decision\s*\(", "make_autonomous_decision()"),
    (r"\bgenerate_reaction\s*\(", "generate_reaction()"),
    (r"\btransition_narrator\s*\(", "transition_narrator()"),
    (r"\bgenerate_gossip\s*\(", "generate_gossip()"),
    (r"\bcalculate_reputation\s*\(", "calculate_reputation()"),
    (r"\bbeat_narrator\s*\(", "beat_narrator()"),
    (r"\bbuild_canon_record\s*\(", "build_canon_record()"),
    (r"\bgenerate_canon_dialogue\s*\(", "generate_canon_dialogue()"),
]

# Hot-path modules (must NOT contain blocking patterns)
HOT_PATH_PATTERNS = [
    "logic/conflict_system/*.py",
    "logic/roleplay_engine.py",
    "logic/event_system.py",
    "logic/fully_integrated_npc_system.py",
    "logic/dynamic_relationships.py",
    "routes/**/*.py",
    "main.py",
]

# Allowed locations (CAN contain blocking patterns)
ALLOWED_PATTERNS = [
    "nyx/tasks/**/*.py",
    "tasks.py",
    "celery_tasks/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "db/migrations/**/*.py",
    "*_hotpath.py",  # Our new hotpath helper modules are allowed (they import but don't call)
]


def match_patterns(file_path: Path, patterns: List[str]) -> bool:
    """Check if file path matches any of the glob patterns."""
    for pattern in patterns:
        if file_path.match(pattern):
            return True
    return False


def is_hot_path_file(file_path: Path) -> bool:
    """Check if this is a hot-path file that should not have blocking calls."""
    if not match_patterns(file_path, HOT_PATH_PATTERNS):
        return False
    # Exclude if it matches allowed patterns
    if match_patterns(file_path, ALLOWED_PATTERNS):
        return False
    return True


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for blocking patterns.

    Returns:
        List of (line_number, pattern_name, line_content) tuples
    """
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Skip function definitions (these are OK - they define the functions used by background tasks)
            if re.match(r"^\s*(async\s+)?def\s+\w+", line):
                continue

            # Skip property definitions
            if re.match(r"^\s*@property", line) or re.match(r"^\s*def\s+\w+\s*\(self\)\s*->\s*Agent:", line):
                continue

            for pattern_re, pattern_name in BLOCKING_PATTERNS:
                if re.search(pattern_re, line):
                    violations.append((line_num, pattern_name, line.strip()))

    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)

    return violations


def main() -> int:
    """Main entry point."""
    repo_root = Path.cwd()
    print(f"🔍 Scanning hot-path modules for blocking LLM calls in {repo_root}")
    print(f"   Hot-path patterns: {HOT_PATH_PATTERNS}")
    print(f"   Allowed patterns: {ALLOWED_PATTERNS}")
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

        # Check if this is a hot-path file
        if not is_hot_path_file(rel_path):
            continue

        file_count += 1
        violations = scan_file(py_file)

        if violations:
            all_violations[rel_path] = violations

    # Report results
    if not all_violations:
        print(f"✅ SUCCESS: No blocking LLM calls found in {file_count} hot-path files")
        print()
        print("All hot-path code is clean! 🎉")
        return 0

    # Print violations
    print(f"❌ FAILURE: Found blocking LLM calls in {len(all_violations)} hot-path files:\n")

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
    print("   1. Move blocking LLM calls to nyx/tasks/background/*.py")
    print("   2. Update hot-path code to dispatch Celery tasks instead")
    print("   3. Use fast helper functions from *_hotpath.py modules")
    print("   4. See docs/hot_path_blockers.md for detailed refactoring guide")
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
