#!/usr/bin/env python3
"""Log aggregate prototype evidence to a local MLflow store."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "readiness": PROJECT_ROOT / "reports/environments/runtime_readiness.json",
    "validation": PROJECT_ROOT / "reports/evaluation/recognition_validation.json",
    "test": PROJECT_ROOT / "reports/evaluation/recognition_test.json",
    "threshold": PROJECT_ROOT / "reports/evaluation/recognition_threshold.json",
    "benchmark": PROJECT_ROOT / "reports/benchmarks/prototype_camera.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(name: str, report: dict) -> dict[str, float]:
    values = report.get("metrics", {})
    return {
        f"{name}_{key}": float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri", default=f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    parser.add_argument("--experiment", default="faceguard-prototype")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; use .venv-mlops/bin/python", file=sys.stderr)
        return 2
    missing = [str(path) for path in REPORTS.values() if not path.is_file()]
    if missing:
        print(f"missing evidence: {missing}", file=sys.stderr)
        return 2

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/runtime.yaml").read_text(encoding="utf-8")
    )
    face, pad, runtime = config["face"], config["pad"], config["runtime"]
    provider = config["providers"][0]
    readiness = _load(REPORTS["readiness"])
    validation = _load(REPORTS["validation"])
    test = _load(REPORTS["test"])
    benchmark = _load(REPORTS["benchmark"])
    session = benchmark.get("session", {})
    target = benchmark.get("target", {})

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("prototype-%Y%m%dT%H%M%SZ")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "faceguard-ai",
                "stage": "prototype",
                "release_id": config["release_id"],
            }
        )
        mlflow.log_params(
            {
                "release_id": config["release_id"],
                "provider": provider,
                "face_pack": face["pack"],
                "detector_sha256": face["detector_sha256"],
                "recognizer_sha256": face["recognizer_sha256"],
                "pad_sha256": pad["sha256"],
                "gallery_sha256": face["gallery_sha256"],
                "recognition_threshold": face["threshold"],
                "pad_threshold": pad["threshold"],
                "pad_min_observations": runtime["pad_min_observations"],
                "gallery_vectors": readiness.get("gallery_vectors"),
                "gallery_identities": readiness.get("gallery_identities"),
                "benchmark_os": target.get("os"),
                "benchmark_cpu": target.get("cpu"),
                "benchmark_camera": str(benchmark.get("source", {}).get("camera")),
            }
        )
        mlflow.log_metrics(
            {
                **_metrics("validation", validation),
                **_metrics("test", test),
                "benchmark_median_latency_ms": float(session.get("median_latency_ms", 0.0)),
                "benchmark_p95_latency_ms": float(session.get("p95_latency_ms", 0.0)),
                "benchmark_processed_fps": float(session.get("processed_fps", 0.0)),
                "benchmark_peak_memory_mb": float(session.get("peak_memory_mb", 0.0)),
                "benchmark_processed_frames": float(session.get("processed_frames", 0.0)),
                "benchmark_dropped_frames": float(session.get("dropped_frames", 0.0)),
            }
        )
        for name, path in REPORTS.items():
            mlflow.log_artifact(str(path), artifact_path="reports")
        mlflow.log_artifact(str(PROJECT_ROOT / "configs/runtime.yaml"), artifact_path="config")
        print(json.dumps({"event": "mlflow_run", "run_id": run.info.run_id, "status": run.info.status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
