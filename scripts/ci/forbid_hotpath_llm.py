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

import ast
import io
import os
import sys
import tokenize
from pathlib import Path
from typing import List, Tuple

# Description used when reporting Runner.run violations
RUNNER_PATTERN_NAME = "Runner.run call"
FORBIDDEN_SNIPPET = "Runner.run("

# Allowed path prefixes (relative to repo root) where Runner.run() is legitimate
ALLOWED_PREFIXES = (
    Path("nyx/gateway"),
    Path("tests"),
)


def is_allowed_path(rel_path: Path) -> bool:
    parts = rel_path.parts

    # nyx/gateway/**
    if parts[:2] == ("nyx", "gateway"):
        return True
    # nyx/core/**
    if parts[:2] == ("nyx", "core"):
        return True
    # tests/**
    if parts and parts[0] == "tests":
        return True
    # any nyx/**/workers/**
    if "nyx" in parts and "workers" in parts[parts.index("nyx") + 1:]:
        return True

    return False



def _collect_docstring_violations(source: str, lines: List[str]) -> List[Tuple[int, str, str]]:
    """Return violations for docstrings containing the forbidden snippet."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: List[Tuple[int, str, str]] = []

    def _check_node(node: ast.AST, kind: str) -> None:
        doc_node = None
        if isinstance(node, ast.Module):
            if node.body and isinstance(node.body[0], ast.Expr):
                doc_node = node.body[0]
        elif node.body and isinstance(node.body[0], ast.Expr):  # type: ignore[attr-defined]
            doc_node = node.body[0]
        if (
            isinstance(doc_node, ast.Expr)
            and isinstance(doc_node.value, ast.Constant)
            and isinstance(doc_node.value.value, str)
        ):
            if FORBIDDEN_SNIPPET in doc_node.value.value:
                lineno = getattr(doc_node.value, "lineno", doc_node.lineno)
                snippet = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else doc_node.value.value
                violations.append((lineno, kind, snippet))

    _check_node(tree, "Runner.run mention in module docstring")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check_node(node, "Runner.run mention in docstring")

    return violations


def _collect_fstring_violations(source: str, lines: List[str]) -> List[Tuple[int, str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: List[Tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            literal_parts = [
                value.value
                for value in node.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            if literal_parts and FORBIDDEN_SNIPPET in "".join(literal_parts):
                lineno = getattr(node, "lineno", 1)
                snippet = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else "Runner.run f-string"
                violations.append((lineno, "Runner.run mention in f-string", snippet))

    return violations


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for forbidden Runner.run usages.

    The scanner tokenizes the file so that string literals and comments are
    ignored. Only actual code tokens (NAME/OP) are considered when detecting the
    `Runner.run` call pattern.

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

    violations.extend(_collect_docstring_violations(source, lines))
    violations.extend(_collect_fstring_violations(source, lines))

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

