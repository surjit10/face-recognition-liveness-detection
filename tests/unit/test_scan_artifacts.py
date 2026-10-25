import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_scan_artifacts", PROJECT_ROOT / "scripts" / "scan_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


def test_restricted_suffix_and_private_prefix_are_reported(tmp_path: Path) -> None:
    (tmp_path / "models/private").mkdir(parents=True)
    (tmp_path / "models/private/model.onnx").write_bytes(b"x")
    (tmp_path / "photo.PNG").write_bytes(b"x")

    findings = SCAN.scan(tmp_path, ["models/private/model.onnx", "photo.PNG"])

    assert len(findings) == 2
    assert all(value.startswith("restricted path:") for value in findings)


def test_credential_shaped_values_are_reported(tmp_path: Path) -> None:
    credential_line = f"{'api' + '_key'}: \"abcdef1234567890\"\n"
    (tmp_path / "config.yaml").write_text(credential_line, encoding="utf-8")
    (tmp_path / "notes.md").write_text("No assignment here: password policy.\n", encoding="utf-8")

    findings = SCAN.scan(tmp_path, ["config.yaml", "notes.md"])

    assert findings == ["credential-shaped value: config.yaml:1"]


def test_clean_public_files_pass(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text(
        "Hashes only, no biometric values.\n", encoding="utf-8"
    )

    assert SCAN.scan(tmp_path, ["docs/README.md", "missing.txt"]) == []
