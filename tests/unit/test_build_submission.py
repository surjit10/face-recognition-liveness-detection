import importlib.util
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_build_submission", PROJECT_ROOT / "scripts" / "build_submission.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_selector_keeps_public_files_and_excludes_local_zones() -> None:
    paths = [
        "README.md",
        "src/faceguard/pipeline.py",
        "reports/benchmarks/prototype_camera.json",
        "reports/environments/environment_local_runtime.json",
        "reports/phase_receipts/promotion_faceguard-v2-development.json",
        "models/private/pad.onnx",
        "models/private/.gitkeep",
        "data/processed/person/face.jpg",
        ".venv/bin/python",
        "reference/README.md",
        "reference/archive/legacy.py",
        "dist/old.zip",
    ]

    selected = BUILD.select_submission_paths(paths)

    assert "README.md" in selected
    assert "src/faceguard/pipeline.py" in selected
    assert "reports/benchmarks/prototype_camera.json" in selected
    assert "reports/environments/environment_local_runtime.json" not in selected
    assert "reports/phase_receipts/promotion_faceguard-v2-development.json" not in selected
    assert "models/private/.gitkeep" in selected
    assert "models/private/pad.onnx" not in selected
    assert "data/processed/person/face.jpg" not in selected
    assert "reference/README.md" not in selected
    assert "dist/old.zip" not in selected


def test_archive_contains_manifest_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("Academic prototype\n", encoding="utf-8")
    (tmp_path / "docs" / "README.md").write_text("Documentation\n", encoding="utf-8")
    output = tmp_path / "dist" / "submission.zip"

    manifest = BUILD.build_archive(
        tmp_path, output, ["README.md", "docs/README.md"]
    )

    assert manifest["file_count"] == 2
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        embedded = json.loads(archive.read("SUBMISSION_MANIFEST.json"))
    assert names == ["README.md", "docs/README.md", "SUBMISSION_MANIFEST.json"]
    assert embedded["file_count"] == 2
    assert all(len(entry["sha256"]) == 64 for entry in embedded["files"])
