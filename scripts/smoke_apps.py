"""Launch every built application briefly and verify its HTML and OpenAPI/health surface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PYTHON_PROJECTS = [path for path in sorted(ROOT.iterdir()) if path.is_dir() and path.name[:2].isdigit() and path.name != "09_flowforge_dag_engine"]


def wait_for(url: str, seconds: float = 12) -> bytes:
    deadline = time.monotonic() + seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return response.read()
        except (URLError, OSError) as error:
            last_error = error
            time.sleep(.2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def hidden_process(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)


def check_python(project: Path) -> None:
    port = 8000 + int(project.name[:2])
    process = hidden_process([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)], project)
    try:
        html = wait_for(f"http://127.0.0.1:{port}/")
        schema = json.loads(wait_for(f"http://127.0.0.1:{port}/openapi.json"))
        if b"<!doctype html>" not in html.lower() or not schema.get("paths"):
            raise RuntimeError("Dashboard or OpenAPI contract is empty")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def check_typescript() -> None:
    project = ROOT / "09_flowforge_dag_engine"
    process = hidden_process(["node", "dist/server.js"], project)
    try:
        html = wait_for("http://127.0.0.1:8009/")
        health = json.loads(wait_for("http://127.0.0.1:8009/api/health"))
        if b"FlowForge" not in html or health.get("status") != "ok":
            raise RuntimeError("FlowForge dashboard or health response is invalid")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    failures: list[str] = []
    for project in PYTHON_PROJECTS:
        try:
            check_python(project)
            print(f"PASS {project.name}")
        except Exception as error:  # Each failure is reported; remaining apps still run.
            failures.append(f"{project.name}: {error}")
            print(f"FAIL {project.name}: {error}")
    try:
        check_typescript()
        print("PASS 09_flowforge_dag_engine")
    except Exception as error:
        failures.append(f"09_flowforge_dag_engine: {error}")
        print(f"FAIL 09_flowforge_dag_engine: {error}")
    if failures:
        raise SystemExit("Smoke test failures:\n" + "\n".join(failures))
    print("All 16 application surfaces passed HTTP smoke testing.")
