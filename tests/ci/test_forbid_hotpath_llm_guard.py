import textwrap
from pathlib import Path

import pytest

from scripts.ci import forbid_hotpath_llm


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


@pytest.fixture
def guard_repo(tmp_path: Path) -> Path:
    # Allowed occurrences inside gateway/core/tests directories.
    _write(
        tmp_path / "nyx/gateway/allow.py",
        """\
        from agents import Runner

        def ok():
            return Runner.run("allowed")
        """,
    )
    _write(
        tmp_path / "nyx/core/allow.py",
        """\
        from agents import Runner

        Runner.run("core")
        """,
    )
    _write(
        tmp_path / "tests/test_allow.py",
        """\
        from agents import Runner

        def test_ok():
            Runner.run("test")
        """,
    )
    return tmp_path


def test_guard_passes_when_only_allowlisted_paths_use_runner(guard_repo: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(guard_repo)
    exit_code = forbid_hotpath_llm.main()
    out, err = capfd.readouterr()
    assert exit_code == 0
    assert "SUCCESS" in out
    assert "Runner.run" not in err


def test_guard_fails_when_disallowed_path_uses_runner(guard_repo: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    disallowed_file = guard_repo / "routes/bad.py"
    _write(
        disallowed_file,
        """\
        from agents import Runner

        def bad():
            Runner.run("nope")
        """,
    )
    monkeypatch.chdir(guard_repo)
    exit_code = forbid_hotpath_llm.main()
    out, err = capfd.readouterr()
    assert exit_code == 1
    assert "routes/bad.py" in out
    assert "Runner.run" in out


def test_guard_ignores_docstring_references(guard_repo: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]) -> None:
    docstring_file = guard_repo / "routes/docstring_reference.py"
    _write(
        docstring_file,
        """\
        '''This docstring mentions Runner.run but should not count as a violation.'''

        def reference():
            '''Docstring again referencing Runner.run without invoking it.'''
            return True
        """,
    )

    monkeypatch.chdir(guard_repo)
    exit_code = forbid_hotpath_llm.main()
    out, err = capfd.readouterr()

    assert exit_code == 0
    assert "docstring_reference.py" not in out
    assert "Runner.run" not in err
