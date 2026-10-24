#!/usr/bin/env python3
"""Extract private ArcFace embeddings from a checked image manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import ConfigurationError  # noqa: E402
from faceguard.extraction import ExtractionError, extract_embeddings  # noqa: E402
from faceguard.models import ModelPackError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", choices=("local", "kaggle"), required=True)
    parser.add_argument("--mode", choices=("audit", "enrollment"), default="audit")
    parser.add_argument("--pack", default="buffalo_m")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Explicit ordered ONNX Runtime providers; no silent provider substitution.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-non-commercial-model-terms",
        action="store_true",
        help="Required for the InsightFace-supplied buffalo_m research model.",
    )
    args = parser.parse_args()
    if not args.acknowledge_non_commercial_model_terms:
        parser.exit(
            2,
            "blocked: buffalo_m supplied weights are non-commercial research only; "
            "review the terms and pass --acknowledge-non-commercial-model-terms\n",
        )
    try:
        report = extract_embeddings(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            decisions_path=args.decisions,
            model_root=args.model_root,
            output_dir=args.output_dir,
            processing_context=args.context,
            mode=args.mode,
            providers=tuple(args.providers),
            pack=args.pack,
            overwrite=args.overwrite,
        )
    except (ConfigurationError, ExtractionError, ModelPackError) as exc:
        parser.exit(2, f"extraction blocked: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
