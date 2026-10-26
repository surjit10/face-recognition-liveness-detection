import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from faceguard.checkins import CheckInError, CheckInStore


def test_verified_checkins_are_deduplicated_and_listed(tmp_path: Path) -> None:
    store = CheckInStore(tmp_path / "private" / "checkins.sqlite3")
    now = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)

    first = store.record_verified(
        "person-a", app_version="0.1.0", model_version="model-a", occurred_at=now
    )
    duplicate = store.record_verified(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=now + timedelta(seconds=10),
    )
    later = store.record_verified(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=now + timedelta(seconds=31),
    )

    assert first is not None
    assert duplicate is None
    assert later is not None
    assert [event.event_id for event in store.recent()] == [later.event_id, first.event_id]
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_export_and_confirmed_clear_primitives(tmp_path: Path) -> None:
    store = CheckInStore(tmp_path / "checkins.sqlite3")
    store.record_verified("person-a", app_version="0.1.0", model_version="model-a")
    target = tmp_path / "export.csv"

    assert store.export_csv(target) == 1
    rendered = target.read_text(encoding="utf-8")
    assert "identity_reference" in rendered
    assert "person-a" in rendered
    assert "local_date" in rendered
    assert "embedding" not in rendered
    assert store.clear() == 1
    assert store.recent() == ()


def test_history_rejects_invalid_or_ambiguous_values(tmp_path: Path) -> None:
    store = CheckInStore(tmp_path / "checkins.sqlite3")
    with pytest.raises(CheckInError):
        store.record_verified("", app_version="0.1.0", model_version="model-a")
    with pytest.raises(CheckInError, match="timezone"):
        store.record_verified(
            "person-a",
            app_version="0.1.0",
            model_version="model-a",
            occurred_at=datetime(2026, 1, 1),
        )


def test_attendance_is_automatic_once_per_local_day(tmp_path: Path) -> None:
    store = CheckInStore(tmp_path / "attendance.sqlite3")
    morning = datetime(2026, 10, 26, 3, 0, tzinfo=timezone.utc)
    evening = datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)

    first = store.record_attendance(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=morning,
        local_date="2026-10-26",
    )
    duplicate = store.record_attendance(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=evening,
        local_date="2026-10-26",
    )
    next_day = store.record_attendance(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=evening + timedelta(days=1),
        local_date="2026-10-27",
    )

    assert first is not None and first.event_type == "attendance"
    assert first.local_date == "2026-10-26"
    assert duplicate is None
    assert next_day is not None and next_day.local_date == "2026-10-27"


def test_existing_history_is_migrated_with_a_local_date(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE checkin_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_reference TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                app_version TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO checkin_events (
                identity_reference, occurred_at, event_type, outcome,
                app_version, model_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "person-a",
                "2026-10-25T18:30:00+00:00",
                "check-in",
                "verified",
                "0.1.0",
                "model-a",
            ),
        )

    events = CheckInStore(path).recent()

    assert len(events) == 1
    assert events[0].local_date == "2026-10-25"


def test_selected_attendance_record_can_be_deleted(tmp_path: Path) -> None:
    store = CheckInStore(tmp_path / "attendance.sqlite3")
    event = store.record_attendance(
        "person-a",
        app_version="0.1.0",
        model_version="model-a",
        occurred_at=datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc),
    )
    assert event is not None

    assert store.delete_event(event.event_id) is True
    assert store.delete_event(event.event_id) is False
    assert store.recent() == ()

    with pytest.raises(CheckInError, match="positive integer"):
        store.delete_event(0)


def test_history_store_has_no_fabrication_or_edit_methods() -> None:
    store = CheckInStore("dummy.sqlite3")
    for forbidden in ("edit_event", "update_event", "modify_event", "set_identity", "set_date"):
        assert not hasattr(store, forbidden), f"CheckInStore must not expose {forbidden}"

