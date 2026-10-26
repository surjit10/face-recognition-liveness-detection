"""Optional local operator PIN verification for protected desktop actions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigurationError, load_yaml_mapping


OPERATOR_ROLES = frozenset({"operator", "attendance_admin"})


@dataclass(frozen=True)
class OperatorAuth:
    """A local session gate; a missing private config keeps supervised mode usable."""

    salt: bytes | None = None
    digest: bytes | None = None
    iterations: int = 300_000
    role: str = "operator"

    @property
    def enabled(self) -> bool:
        return self.salt is not None and self.digest is not None

    def verify(self, pin: str) -> bool:
        if not self.enabled:
            return True
        assert self.salt is not None and self.digest is not None
        candidate = hashlib.pbkdf2_hmac(
            "sha256", pin.encode("utf-8"), self.salt, self.iterations
        )
        return hmac.compare_digest(candidate, self.digest)

    def permits(self, page: str) -> bool:
        """Return whether this credential role may open an application page."""
        if not self.enabled or self.role == "operator":
            return True
        return page in {"verify", "history"}


def build_operator_document(
    pin: str,
    *,
    salt: bytes | None = None,
    iterations: int = 300_000,
    role: str = "attendance_admin",
) -> dict[str, object]:
    if len(pin) < 4 or len(pin) > 64 or any(ord(char) < 32 for char in pin):
        raise ConfigurationError("Operator PIN must contain 4 to 64 printable characters")
    if iterations < 100_000 or iterations > 2_000_000:
        raise ConfigurationError("Operator PIN iterations must be within 100000..2000000")
    if role not in OPERATOR_ROLES:
        raise ConfigurationError(f"Operator role must be one of {sorted(OPERATOR_ROLES)}")
    resolved_salt = salt or secrets.token_bytes(16)
    if len(resolved_salt) < 16:
        raise ConfigurationError("Operator PIN salt must contain at least 16 bytes")
    digest = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), resolved_salt, iterations
    )
    return {
        "schema_version": 2,
        "role": role,
        "algorithm": "pbkdf2_sha256",
        "iterations": iterations,
        "salt_hex": resolved_salt.hex(),
        "pin_hash_hex": digest.hex(),
    }


def load_operator_auth(path: str | Path | None) -> OperatorAuth:
    if path is None or not Path(path).is_file():
        return OperatorAuth()
    document = load_yaml_mapping(path)
    base_keys = {
        "schema_version",
        "algorithm",
        "iterations",
        "salt_hex",
        "pin_hash_hex",
    }
    schema_version = document.get("schema_version")
    expected = base_keys if schema_version == 1 else base_keys | {"role"}
    if set(document) != expected:
        raise ConfigurationError(
            "Operator config keys mismatch; "
            f"missing={sorted(expected - set(document))}, unknown={sorted(set(document) - expected)}"
        )
    if schema_version not in {1, 2}:
        raise ConfigurationError("Operator config schema_version must be 1 or 2")
    if document["algorithm"] != "pbkdf2_sha256":
        raise ConfigurationError("Operator config algorithm must be pbkdf2_sha256")
    iterations = document["iterations"]
    if not isinstance(iterations, int) or not 100_000 <= iterations <= 2_000_000:
        raise ConfigurationError("Operator PIN iterations must be within 100000..2000000")
    try:
        salt = bytes.fromhex(str(document["salt_hex"]))
        digest = bytes.fromhex(str(document["pin_hash_hex"]))
    except ValueError as exc:
        raise ConfigurationError("Operator salt and PIN hash must be hexadecimal") from exc
    if len(salt) < 16 or len(digest) != 32:
        raise ConfigurationError("Operator salt/hash lengths are invalid")
    role = "operator" if schema_version == 1 else document["role"]
    if not isinstance(role, str) or role not in OPERATOR_ROLES:
        raise ConfigurationError(f"Operator role must be one of {sorted(OPERATOR_ROLES)}")
    return OperatorAuth(salt=salt, digest=digest, iterations=iterations, role=role)
