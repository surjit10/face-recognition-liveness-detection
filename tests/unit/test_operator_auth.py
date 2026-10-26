from pathlib import Path

import pytest

from faceguard.config import ConfigurationError
import yaml

from faceguard.operator_auth import build_operator_document, load_operator_auth


def test_missing_operator_config_allows_supervised_mode(tmp_path: Path) -> None:
    auth = load_operator_auth(tmp_path / "missing.yaml")
    assert auth.enabled is False
    assert auth.verify("") is True


def test_operator_pin_is_verified_without_storing_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "operator.yaml"
    document = build_operator_document("2468", salt=b"0123456789abcdef")
    rendered = yaml.safe_dump(document, sort_keys=False)
    path.write_text(rendered, encoding="utf-8")
    auth = load_operator_auth(path)
    assert auth.enabled is True
    assert auth.role == "attendance_admin"
    assert auth.verify("2468") is True
    assert auth.verify("0000") is False
    assert auth.permits("history") is True
    assert auth.permits("users") is False


def test_operator_config_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "operator.yaml"
    path.write_text("schema_version: 1\npin_sha256: bad\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_operator_auth(path)


def test_operator_pin_policy_rejects_short_values() -> None:
    with pytest.raises(ConfigurationError):
        build_operator_document("123")


def test_full_operator_role_can_access_all_pages(tmp_path: Path) -> None:
    path = tmp_path / "operator.yaml"
    document = build_operator_document(
        "2468", salt=b"0123456789abcdef", role="operator"
    )
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    auth = load_operator_auth(path)

    assert auth.role == "operator"
    assert auth.permits("enroll") is True
    assert auth.permits("users") is True
    assert auth.permits("settings") is True


def test_legacy_operator_document_keeps_full_operator_role(tmp_path: Path) -> None:
    path = tmp_path / "operator.yaml"
    document = build_operator_document(
        "2468", salt=b"0123456789abcdef", role="operator"
    )
    document["schema_version"] = 1
    document.pop("role")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    auth = load_operator_auth(path)

    assert auth.role == "operator"
    assert auth.permits("users") is True


def test_unknown_operator_role_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="role"):
        build_operator_document("2468", role="unknown")


def test_attendance_admin_role_permits_only_verify_and_history(tmp_path: Path) -> None:
    path = tmp_path / "operator.yaml"
    document = build_operator_document(
        "2468", salt=b"0123456789abcdef", role="attendance_admin"
    )
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    auth = load_operator_auth(path)

    assert auth.role == "attendance_admin"
    assert auth.permits("verify") is True
    assert auth.permits("history") is True
    assert auth.permits("enroll") is False
    assert auth.permits("users") is False
    assert auth.permits("settings") is False


def test_configure_operator_cli_generates_valid_private_config(monkeypatch, tmp_path: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "faceguard_configure_operator",
        Path(__file__).resolve().parents[2] / "scripts" / "configure_operator.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_path = tmp_path / "admin_op.yaml"
    monkeypatch.setattr(mod.getpass, "getpass", lambda prompt: "secret1234")
    assert mod.main(["--output", str(out_path)]) == 0

    assert out_path.is_file()
    assert out_path.stat().st_mode & 0o777 == 0o600
    auth = load_operator_auth(out_path)
    assert auth.role == "attendance_admin"
    assert auth.verify("secret1234") is True
    assert auth.permits("history") is True
    assert auth.permits("settings") is False

    # Test operator role
    op_path = tmp_path / "operator.yaml"
    assert mod.main(["--output", str(op_path), "--role", "operator"]) == 0
    op_auth = load_operator_auth(op_path)
    assert op_auth.role == "operator"
    assert op_auth.permits("settings") is True


def test_configure_operator_cli_blocks_mismatched_pin(monkeypatch, tmp_path: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "faceguard_configure_operator",
        Path(__file__).resolve().parents[2] / "scripts" / "configure_operator.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_path = tmp_path / "failed.yaml"
    prompts = iter(["secret1234", "mismatch9999"])
    monkeypatch.setattr(mod.getpass, "getpass", lambda prompt: next(prompts))

    assert mod.main(["--output", str(out_path)]) == 2
    assert not out_path.exists()

