"""Capture a reproducible 1440×1100 dashboard screenshot for every project."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PROJECTS = [path for path in sorted(ROOT.iterdir()) if path.is_dir() and path.name[:2].isdigit()]


def wait_for(url: str, seconds: float = 15) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            time.sleep(.25)
    raise RuntimeError(f"Timed out waiting for {url}")


def process(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)


def capture(project: Path) -> Path:
    number = int(project.name[:2])
    port = 8000 + number
    if number == 9:
        server = process(["node", "dist/server.js"], project)
    else:
        server = process([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)], project)
    destination = project / "docs" / "screenshots" / "dashboard.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        wait_for(f"http://127.0.0.1:{port}/")
        completed = subprocess.run(
            [
                str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--window-size=1440,1100", "--virtual-time-budget=5000",
                f"--screenshot={destination}", f"http://127.0.0.1:{port}/",
            ],
            cwd=project,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode != 0 or not destination.exists():
            raise RuntimeError(completed.stdout.decode(errors="replace"))
        return destination
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    if not CHROME.exists():
        raise SystemExit(f"Chrome not found at {CHROME}")
    for item in PROJECTS:
        screenshot = capture(item)
        print(f"CAPTURED {screenshot.relative_to(ROOT)}", flush=True)
    print(f"Captured {len(PROJECTS)} dashboard screenshots.")
