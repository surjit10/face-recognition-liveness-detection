"""Privacy-limited local check-in history with duplicate suppression."""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


class CheckInError(RuntimeError):
    """A check-in record or database operation is invalid."""


@dataclass(frozen=True)
class CheckInEvent:
    event_id: int
    identity_reference: str
    occurred_at: str
    local_date: str
    event_type: str
    outcome: str
    app_version: str
    model_version: str


def _safe_text(value: str, field: str, *, max_length: int = 128) -> str:
    clean = value.strip()
    if not clean or len(clean) > max_length or any(ord(char) < 32 for char in clean):
        raise CheckInError(f"{field} must be non-empty printable text up to {max_length} characters")
    return clean


class CheckInStore:
    """Open short-lived SQLite connections so UI workers can use the store safely."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkin_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        identity_reference TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        local_date TEXT,
                        event_type TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK (outcome = 'verified'),
                        app_version TEXT NOT NULL,
                        model_version TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(checkin_events)")
                }
                if "local_date" not in columns:
                    connection.execute("ALTER TABLE checkin_events ADD COLUMN local_date TEXT")
                connection.execute(
                    """
                    UPDATE checkin_events SET local_date = substr(occurred_at, 1, 10)
                    WHERE local_date IS NULL OR local_date = ''
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_checkin_recent
                    ON checkin_events(identity_reference, event_type, occurred_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_attendance_day
                    ON checkin_events(local_date DESC, identity_reference, event_type)
                    """
                )
                connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise CheckInError(f"Unable to initialize check-in history: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def record_verified(
        self,
        identity_reference: str,
        *,
        event_type: str = "check-in",
        app_version: str,
        model_version: str,
        occurred_at: datetime | None = None,
        dedupe_seconds: int = 30,
        local_date: str | None = None,
    ) -> CheckInEvent | None:
        """Record one verified event, or return None when it is a recent duplicate."""
        identity = _safe_text(identity_reference, "identity reference")
        kind = _safe_text(event_type, "event type", max_length=32)
        application = _safe_text(app_version, "application version", max_length=64)
        model = _safe_text(model_version, "model version", max_length=128)
        if dedupe_seconds < 0:
            raise CheckInError("dedupe_seconds must not be negative")
        moment = occurred_at or datetime.now().astimezone()
        if moment.tzinfo is None:
            raise CheckInError("occurred_at must include a timezone")
        resolved_date = local_date or moment.date().isoformat()
        try:
            date.fromisoformat(resolved_date)
        except ValueError as exc:
            raise CheckInError("local_date must use YYYY-MM-DD") from exc
        utc_moment = moment.astimezone(timezone.utc)
        timestamp = utc_moment.isoformat()
        cutoff = (utc_moment - timedelta(seconds=dedupe_seconds)).isoformat()
        self.initialize()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                duplicate = connection.execute(
                    """
                    SELECT event_id FROM checkin_events
                    WHERE identity_reference = ? AND event_type = ?
                      AND outcome = 'verified' AND occurred_at >= ?
                    ORDER BY occurred_at DESC LIMIT 1
                    """,
                    (identity, kind, cutoff),
                ).fetchone()
                if duplicate is not None:
                    return None
                cursor = connection.execute(
                    """
                    INSERT INTO checkin_events (
                        identity_reference, occurred_at, local_date, event_type, outcome,
                        app_version, model_version
                    ) VALUES (?, ?, ?, ?, 'verified', ?, ?)
                    """,
                    (identity, timestamp, resolved_date, kind, application, model),
                )
                event_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise CheckInError(f"Unable to record check-in: {exc}") from exc
        return CheckInEvent(
            event_id,
            identity,
            timestamp,
            resolved_date,
            kind,
            "verified",
            application,
            model,
        )

    def record_attendance(
        self,
        identity_reference: str,
        *,
        app_version: str,
        model_version: str,
        occurred_at: datetime | None = None,
        local_date: str | None = None,
    ) -> CheckInEvent | None:
        """Mark attendance once per identity and local calendar day."""
        identity = _safe_text(identity_reference, "identity reference")
        application = _safe_text(app_version, "application version", max_length=64)
        model = _safe_text(model_version, "model version", max_length=128)
        moment = occurred_at or datetime.now().astimezone()
        if moment.tzinfo is None:
            raise CheckInError("occurred_at must include a timezone")
        resolved_date = local_date or moment.date().isoformat()
        try:
            date.fromisoformat(resolved_date)
        except ValueError as exc:
            raise CheckInError("local_date must use YYYY-MM-DD") from exc
        timestamp = moment.astimezone(timezone.utc).isoformat()
        self.initialize()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                duplicate = connection.execute(
                    """
                    SELECT event_id FROM checkin_events
                    WHERE identity_reference = ? AND event_type = 'attendance'
                      AND outcome = 'verified' AND local_date = ?
                    LIMIT 1
                    """,
                    (identity, resolved_date),
                ).fetchone()
                if duplicate is not None:
                    return None
                cursor = connection.execute(
                    """
                    INSERT INTO checkin_events (
                        identity_reference, occurred_at, local_date, event_type, outcome,
                        app_version, model_version
                    ) VALUES (?, ?, ?, 'attendance', 'verified', ?, ?)
                    """,
                    (identity, timestamp, resolved_date, application, model),
                )
                event_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise CheckInError(f"Unable to record attendance: {exc}") from exc
        return CheckInEvent(
            event_id,
            identity,
            timestamp,
            resolved_date,
            "attendance",
            "verified",
            application,
            model,
        )

    def recent(self, *, limit: int = 200) -> tuple[CheckInEvent, ...]:
        if limit <= 0 or limit > 10_000:
            raise CheckInError("history limit must be within 1..10000")
        self.initialize()
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT event_id, identity_reference, occurred_at, local_date, event_type,
                           outcome, app_version, model_version
                    FROM checkin_events ORDER BY occurred_at DESC, event_id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise CheckInError(f"Unable to read check-in history: {exc}") from exc
        return tuple(CheckInEvent(**dict(row)) for row in rows)

    def export_csv(self, output_path: str | Path) -> int:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        events = self.recent(limit=10_000)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=target.parent,
                prefix=f".{target.name}.",
                delete=False,
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "event_id",
                        "identity_reference",
                        "occurred_at",
                        "local_date",
                        "event_type",
                        "outcome",
                        "app_version",
                        "model_version",
                    )
                )
                for event in reversed(events):
                    writer.writerow(
                        (
                            event.event_id,
                            event.identity_reference,
                            event.occurred_at,
                            event.local_date,
                            event.event_type,
                            event.outcome,
                            event.app_version,
                            event.model_version,
                        )
                    )
                temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except OSError as exc:
            raise CheckInError(f"Unable to export check-in history: {exc}") from exc
        return len(events)

    def clear(self) -> int:
        self.initialize()
        try:
            with self._connect() as connection:
                count = int(connection.execute("SELECT COUNT(*) FROM checkin_events").fetchone()[0])
                connection.execute("DELETE FROM checkin_events")
        except sqlite3.Error as exc:
            raise CheckInError(f"Unable to clear check-in history: {exc}") from exc
        return count

    def delete_event(self, event_id: int) -> bool:
        """Delete one selected attendance event; return whether it existed."""
        if not isinstance(event_id, int) or isinstance(event_id, bool) or event_id <= 0:
            raise CheckInError("event_id must be a positive integer")
        self.initialize()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM checkin_events WHERE event_id = ?", (event_id,)
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise CheckInError(f"Unable to delete attendance event: {exc}") from exc
