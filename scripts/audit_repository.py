"""Fast, dependency-free structural audit for the 16-project portfolio."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "00_dynamic_todo_workspace",
    "01_nyc_taxi_trip_prediction",
    "02_nano_llm_transformer",
    "03_customer_segmentation_clustering",
    "04_associative_pattern_mining",
    "05_data_science_skills_lab",
    "06_anomaly_detection",
    "07_automl_autogluon",
    "08_datascience_visual_mastery",
    "09_flowforge_dag_engine",
    "10_crispdm_masters_curriculum",
    "11_enterprise_ds_audit",
    "12_timeseries_forecasting",
    "13_crispdm_nyc_taxi_audit_platform",
    "14_autogluon_multimodal_automl_suite",
    "15_spy_timeseries_sota_forecasting",
]
REQUIRED = ("README.md", "PROMPT.md")


def audit() -> list[str]:
    findings: list[str] = []
    for name in EXPECTED:
        project = ROOT / name
        if not project.is_dir():
            findings.append(f"MISSING project: {name}")
            continue
        for filename in REQUIRED:
            if not (project / filename).is_file():
                findings.append(f"MISSING {name}/{filename}")
        has_runtime = (project / "app.py").is_file() or (project / "package.json").is_file()
        if not has_runtime:
            findings.append(f"MISSING runtime: {name}")
        has_test = any(project.glob("test*.py")) or any(project.glob("**/*.test.ts"))
        if not has_test:
            findings.append(f"MISSING tests: {name}")
    return findings


if __name__ == "__main__":
    issues = audit()
    if issues:
        print("Repository audit failed:")
        print("\n".join(f"- {item}" for item in issues))
        raise SystemExit(1)
    print(f"Repository audit passed: {len(EXPECTED)} independent projects found.")
