"""§19.0 preflight checks — verify environment before Phase 0 work."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def _check(desc: str, fn: Callable[[], tuple[bool, str]]) -> tuple[str, bool, str]:
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"exception: {e}"
    return desc, ok, detail


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _cmd(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or r.stderr or "").strip().splitlines()[:3]
        return r.returncode == 0, "\n".join(out)
    except Exception as e:
        return False, f"exception: {e}"


def run_preflight(output: str | Path = "PREFLIGHT-REPORT.md") -> bool:
    checks: list[tuple[str, bool, str]] = []

    # apache-gravitino resolves on PyPI
    def check_pkg():
        if not _has_cmd("pip3") and not _has_cmd("pip"):
            return False, "no pip on PATH"
        pip = "pip3" if _has_cmd("pip3") else "pip"
        ok, out = _cmd([pip, "index", "versions", "apache-gravitino"])
        return ok, out

    checks.append(_check("apache-gravitino available on PyPI", check_pkg))

    # import check (best-effort: only if installed)
    def check_import():
        try:
            importlib.import_module("gravitino")
            return True, "import ok"
        except Exception as e:
            return False, f"not installed yet: {e}"

    checks.append(_check("`from gravitino import ...` imports", check_import))

    # docker images
    if _has_cmd("docker"):
        checks.append(_check(
            "docker pull apache/gravitino:1.2.0",
            lambda: _cmd(["docker", "pull", "apache/gravitino:1.2.0"], timeout=120),
        ))
        checks.append(_check(
            "docker pull apache/gravitino-playground:ranger-0.1.0",
            lambda: _cmd(
                ["docker", "pull", "apache/gravitino-playground:ranger-0.1.0"], timeout=120
            ),
        ))
    else:
        checks.append(("docker pull ...", False, "docker not on PATH"))

    # Maven artefacts
    def curl_head(url: str):
        if not _has_cmd("curl"):
            return False, "no curl"
        ok, out = _cmd(["curl", "-sI", url], timeout=15)
        first = out.splitlines()[0] if out else ""
        return "200" in first, first

    checks.append(_check(
        "Gravitino Spark connector JAR reachable",
        lambda: curl_head(
            "https://repo1.maven.org/maven2/org/apache/gravitino/gravitino-spark-connector-runtime-3.4_2.12/1.2.0/gravitino-spark-connector-runtime-3.4_2.12-1.2.0.pom"
        ),
    ))

    # gravitino-playground baseline
    def check_playground():
        tmp = Path("/tmp/gp-preflight")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        ok, _ = _cmd(
            ["git", "clone", "--depth", "1",
             "https://github.com/apache/gravitino-playground", str(tmp)],
            timeout=60,
        )
        if not ok:
            return False, "clone failed"
        return (tmp / "docker-compose.yaml").exists(), "docker-compose.yaml present"

    checks.append(_check("gravitino-playground baseline exists", check_playground))

    # Write report
    out_path = Path(output)
    lines = ["# Preflight report — §19.0", ""]
    all_ok = True
    for desc, ok, detail in checks:
        mark = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        lines.append(f"- {mark} {desc}")
        if detail:
            for l in detail.splitlines():
                lines.append(f"  - {l}")
    lines.append("")
    lines.append(
        "Status: " + ("**all checks passed**" if all_ok else "**FAIL — see entries above**")
    )
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    return all_ok
