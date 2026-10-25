from pathlib import Path

import pytest
import yaml

from faceguard.config import ConfigurationError
from faceguard.runtime_config import load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_candidate_runtime_config_blocks_unresolved_recognition_values() -> None:
    with pytest.raises(ConfigurationError, match="face.threshold"):
        load_runtime_config(
            PROJECT_ROOT / "configs/runtime.yaml", project_root=PROJECT_ROOT
        )


def test_complete_runtime_config_parses_without_silent_defaults(tmp_path: Path) -> None:
    document = yaml.safe_load(
        (PROJECT_ROOT / "configs/runtime.yaml").read_text(encoding="utf-8")
    )
    document["face"]["gallery_sha256"] = "a" * 64
    document["face"]["threshold"] = 0.6
    document["face"]["min_top1_top2_margin"] = 0.1
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_runtime_config(path, project_root=PROJECT_ROOT)
    assert config.pipeline_policy.pad_threshold == pytest.approx(0.7626009583473206)
    assert config.pipeline_policy.recognition_threshold == 0.6
    assert config.frame_queue_size == 2
