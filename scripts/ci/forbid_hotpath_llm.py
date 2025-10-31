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

# Background-only methods (called ONLY from background tasks, not from event handlers)
# These methods are allowed to contain LLM calls even though they're in hot-path files
BACKGROUND_ONLY_METHODS = [
    "create_stakeholder",
    "make_autonomous_decision",
    "generate_reaction",
    "adapt_stakeholder_role",
    "initialize_conflict_flow",
    "update_conflict_flow",
    "generate_dramatic_beat",
    "narrate_phase_transition",
    "_handle_phase_transition",
    "evaluate_for_canon",
    "check_lore_compliance",
    "generate_canon_references",
    "_create_stakeholders_for_npcs",
    "generate_manifestations",
    "narrate_escalation",
    "_generate_tension_manifestation_llm",
    "check_tension_breaking_point",
    "perform_bundle_generation_and_cache",
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

        in_background_method = False
        method_indent = 0
        in_signature = False  # Track if we're still in the method signature
        in_multiline_string = False  # Track if we're in a multi-line string
        string_delimiter = None  # Track the delimiter (""" or ''')

        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines and comments (but not if we're in a string)
            stripped = line.strip()
            if not in_multiline_string and (not stripped or stripped.startswith("#")):
                continue

            # Track multi-line strings (""" or ''')
            # Count occurrences of """ and ''' to detect entry/exit
            if '"""' in line or "'''" in line:
                # Simple heuristic: count triple quotes
                triple_double = line.count('"""')
                triple_single = line.count("'''")

                if triple_double > 0:
                    if not in_multiline_string:
                        in_multiline_string = True
                        string_delimiter = '"""'
                    elif string_delimiter == '"""':
                        # Toggle for each occurrence
                        for _ in range(triple_double):
                            in_multiline_string = not in_multiline_string

                if triple_single > 0:
                    if not in_multiline_string:
                        in_multiline_string = True
                        string_delimiter = "'''"
                    elif string_delimiter == "'''":
                        for _ in range(triple_single):
                            in_multiline_string = not in_multiline_string

            # Skip lines inside multi-line strings when checking indentation
            if in_multiline_string:
                # Still need to check for violations in the actual code part
                if in_background_method:
                    continue  # Skip checking violations in background methods
                # Otherwise continue to check for violations

            # Get current line indentation
            line_indent = len(line) - len(line.lstrip())

            # Check if we're entering a method definition
            method_match = re.match(r"^(\s*)(async\s+)?def\s+(\w+)", line)
            if method_match:
                indent = len(method_match.group(1))
                method_name = method_match.group(3)

                # If we're in a background method and hit a new method at same/less indent, we've exited
                if in_background_method and indent <= method_indent:
                    in_background_method = False
                    in_signature = False

                # Check if this new method is background-only
                if method_name in BACKGROUND_ONLY_METHODS:
                    in_background_method = True
                    method_indent = indent
                    in_signature = True  # We're now in the method signature

                continue

            # Only check indentation-based exit if NOT in a multi-line string
            if not in_multiline_string:
                # If we're past the signature and hit code at same/less indentation as def, we've exited
                if in_background_method and not in_signature and line_indent <= method_indent and stripped:
                    in_background_method = False
                    in_signature = False

            # Check if we've exited the signature (line ends with : which is not in a string)
            if in_background_method and in_signature and stripped.endswith(":"):
                in_signature = False  # Signature complete, now in method body

            # Skip violations inside background-only methods
            if in_background_method:
                continue

            # Skip property definitions
            if re.match(r"^\s*@property", line) or re.match(r"^\s*def\s+\w+\s*\(self\)\s*->\s*Agent:", line):
                continue

            # Check for blocking patterns
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
