"""Run every independent project's unit tests and the TypeScript build."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = [path for path in sorted(ROOT.iterdir()) if path.is_dir() and path.name[:2].isdigit()]


def run(command: list[str], cwd: Path) -> bool:
    completed = subprocess.run(command, cwd=cwd)
    return completed.returncode == 0


if __name__ == "__main__":
    failed: list[str] = []
    for project in PROJECTS:
        print(f"\n=== {project.name} ===", flush=True)
        if project.name.startswith("09_"):
            test_ok = run(["cmd", "/c", "npm", "test"] if sys.platform == "win32" else ["npm", "test"], project)
            build_ok = run(["cmd", "/c", "npm", "run", "build"] if sys.platform == "win32" else ["npm", "run", "build"], project)
            if not (test_ok and build_ok):
                failed.append(project.name)
        elif not run([sys.executable, "-m", "unittest", "-v"], project):
            failed.append(project.name)
    if failed:
        raise SystemExit("Failed projects: " + ", ".join(failed))
    print(f"\nAll {len(PROJECTS)} project test suites passed.")
