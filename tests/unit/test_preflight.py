"""Smoke tests for the preflight runner — all subprocess calls stubbed."""

from pathlib import Path

import spark_ai_functions.preflight as P


def test_run_preflight_writes_report(monkeypatch, tmp_path: Path):
    # Stub every heavy external. Happy-path values are tailored per check.
    def _fake_cmd(cmd, timeout=30):
        if cmd[:2] == ["curl", "-sI"]:
            return True, "HTTP/2 200 ok"
        return True, "ok"

    monkeypatch.setattr(P, "_cmd", _fake_cmd)
    monkeypatch.setattr(P, "_has_cmd", lambda name: True)

    # Make the playground-baseline check see the file.
    monkeypatch.setattr(
        "pathlib.Path.exists",
        lambda self: True,
    )

    import importlib as _il
    monkeypatch.setattr(_il, "import_module", lambda name: object())

    out = tmp_path / "report.md"
    ok = P.run_preflight(output=out)
    assert out.exists()
    body = out.read_text()
    assert "all checks passed" in body or "Preflight report" in body
    assert ok is True


def test_run_preflight_reports_failures(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(P, "_cmd", lambda cmd, timeout=30: (False, "nope"))
    monkeypatch.setattr(P, "_has_cmd", lambda name: False)

    out = tmp_path / "bad.md"
    ok = P.run_preflight(output=out)
    assert ok is False
    body = out.read_text()
    assert "FAIL" in body
