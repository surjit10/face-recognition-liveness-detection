#!/usr/bin/env python3
"""Benchmark the three frozen ONNX stages without retaining biometric inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "face_detection": (
        ROOT / "models/private/insightface/models/buffalo_m/det_2.5g.onnx",
        (1, 3, 640, 640),
    ),
    "pad": (
        ROOT / "models/private/pad_candidate/mobilenetv3_pad.onnx",
        (1, 3, 224, 224),
    ),
    "face_recognition": (
        ROOT / "models/private/insightface/models/buffalo_m/w600k_r50.onnx",
        (1, 3, 112, 112),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_model(
    path: Path, shape: tuple[int, ...], warmup: int, iterations: int, seed: int
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected providers for {path}: {session.get_providers()}")
    input_meta = session.get_inputs()[0]
    rng = np.random.default_rng(seed)
    tensor = rng.standard_normal(shape, dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {input_meta.name: tensor})
    latencies: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outputs = session.run(None, {input_meta.name: tensor})
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if not outputs or any(not np.isfinite(value).all() for value in outputs):
        raise RuntimeError(f"Non-finite or absent output from {path}")
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "model": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "input_name": input_meta.name,
        "input_shape": list(shape),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "latency_ms": {
            "mean": round(float(np.mean(values)), 3),
            "median": round(float(np.median(values)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "maximum": round(float(np.max(values)), 3),
        },
        "throughput_fps_from_mean": round(1000.0 / float(np.mean(values)), 3),
        "providers": session.get_providers(),
        "outputs_finite": True,
    }


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/benchmarks/model_stages_cpu.json",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    stages = {
        name: benchmark_model(path, shape, args.warmup, args.iterations, args.seed)
        for name, (path, shape) in MODELS.items()
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "event": "frozen_onnx_stage_benchmark",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
        },
        "method": {
            "input": "deterministic synthetic finite tensors; no biometric data",
            "scope": "raw ONNX inference only; excludes decode, crop, tracking and UI",
            "seed": args.seed,
        },
        "stages": stages,
        "privacy": {
            "frames_retained": False,
            "embeddings_retained": False,
            "biometric_inputs_used": False,
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
