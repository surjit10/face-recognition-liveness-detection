"""Unified desktop controller for verification, enrollment and local records."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..bootstrap import RuntimeBundle, RuntimeFactory
from ..checkins import CheckInStore
from ..gallery_admin import (
    delete_identity,
    duplicate_candidates,
    enroll_identity,
    inspect_identity,
    list_identities,
)
from ..identity_labels import load_identity_labels
from ..operator_auth import load_operator_auth
from ..presentation import REASON_GUIDANCE
from ..session import SessionOptions, SessionStatus, UiSnapshot, VerificationSession
from .live_view import LiveView
from .pages import EnrollmentView, HistoryView, NavigationShell, SettingsView, UsersView
from .theme import configure_root


class DesktopApp:
    """Coordinate one shared runtime and camera across every application screen."""

    POLL_MS = 40

    def __init__(
        self,
        root: tk.Tk,
        *,
        config_path: Path,
        project_root: Path,
        camera_index: int = 0,
        identity_labels_path: Path | None = None,
        history_db_path: Path | None = None,
        operator_config_path: Path | None = None,
    ) -> None:
        if camera_index != 0:
            raise ValueError("Only authorized camera 0 is supported")
        self.root = root
        self.config_path = config_path
        self.project_root = project_root
        self.camera_index = camera_index
        self.identity_labels = load_identity_labels(identity_labels_path)
        self.operator_auth = load_operator_auth(operator_config_path)
        self._operator_unlocked = not self.operator_auth.enabled
        self._presentation_mode = False
        self.history_store = CheckInStore(
            history_db_path or project_root / "reports/private/checkins.sqlite3"
        )
        self.exit_code = 0
        self._closed = False
        self._bundle: RuntimeBundle | None = None
        self._session: VerificationSession | None = None
        self._session_mode: str | None = None
        self._active_page = "verify"
        self._latest_snapshot: UiSnapshot | None = None
        self._latest_verified: tuple[str, str, int | None] | None = None
        self._checkin_inflight = False
        self._recorded_checkin_key: tuple[str, int | None] | None = None
        self._verification_finishing = False
        self._enrollment_embeddings: list[np.ndarray] = []
        self._duplicate_ids: tuple[str, ...] = ()
        self._sample_inflight = False
        self._readiness_generation = 0
        self._readiness_updates: queue.Queue[tuple[int, bool, Any]] = queue.Queue(maxsize=2)
        self._task_updates: queue.Queue[tuple[str, bool, Any]] = queue.Queue()
        self._auto_reset_timer: str | None = None

        root.title("FaceGuard AI")
        root.geometry("1420x880")
        root.minsize(1120, 720)
        configure_root(root)
        self.shell = NavigationShell(
            root,
            on_navigate=self.show_page,
        )
        self.shell.pack(fill="both", expand=True)
        if self.operator_auth.enabled and self.operator_auth.role == "attendance_admin":
            self.shell.set_visible_pages({"verify", "history"})

        self.view = LiveView(
            self.shell.content,
            on_start=self.start_camera,
            on_stop=self.stop_camera,
            identity_labels=self.identity_labels,
        )
        self.enrollment_view = EnrollmentView(
            self.shell.content,
            on_start=self.start_enrollment_camera,
            on_stop=self.stop_camera,
            on_capture=self.capture_enrollment_sample,
            on_confirm=self.confirm_enrollment,
            on_clear=self.clear_enrollment,
        )
        self.users_view = UsersView(
            self.shell.content,
            on_refresh=self.refresh_users,
            on_inspect=self.inspect_selected_user,
            on_delete=self.delete_selected_user,
        )
        self.history_view = HistoryView(
            self.shell.content,
            on_refresh=self.refresh_history,
            on_export=self.export_history,
            on_delete_selected=self.delete_selected_attendance,
            on_clear=self.clear_history,
            on_lock=self.lock_operator,
        )
        if self.operator_auth.enabled and self.operator_auth.role == "attendance_admin":
            self.history_view.show_admin_lock()
        self.settings_view = SettingsView(
            self.shell.content,
            on_reset_demo=self.reset_demo_session,
            on_lock=self.lock_operator,
        )
        self._pages = {
            "verify": self.view,
            "enroll": self.enrollment_view,
            "users": self.users_view,
            "history": self.history_view,
            "settings": self.settings_view,
        }
        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.shell.select("verify")
        self.view.tkraise()
        self._update_operator_ui()

        shortcuts = ("verify", "enroll", "users", "history", "settings")
        for index, page_name in enumerate(shortcuts, start=1):
            root.bind(
                f"<Control-Key-{index}>",
                lambda _event, target=page_name: self.show_page(target),
            )
        root.bind("<F11>", lambda _event: self.toggle_presentation_mode())
        root.bind("<Escape>", self._leave_presentation_mode)

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(120, self.check_system)
        self._poll_id = root.after(self.POLL_MS, self._poll)

    @property
    def gallery_path(self) -> Path:
        if self._bundle is not None:
            return self._bundle.config.gallery_path
        return self.project_root / "models/private/gallery/gallery.npz"

    def _run_task(self, name: str, operation: Callable[[], Any]) -> None:
        def run() -> None:
            try:
                result = operation()
                update = (name, True, result)
            except Exception as exc:
                update = (name, False, exc)
            self._task_updates.put(update)

        threading.Thread(target=run, name=f"faceguard-{name}", daemon=True).start()

    def _update_operator_ui(self) -> None:
        self.settings_view.set_operator_state(
            configured=self.operator_auth.enabled,
            unlocked=self._operator_unlocked,
        )

    def _ensure_operator(self) -> bool:
        if self._operator_unlocked or not self.operator_auth.enabled:
            return True
        is_attendance_admin = self.operator_auth.role == "attendance_admin"
        title = "Administrator access" if is_attendance_admin else "Operator access"
        prompt = (
            "Enter the local administrator PIN to manage attendance:"
            if is_attendance_admin
            else "Enter the local operator PIN to unlock protected actions:"
        )
        pin = simpledialog.askstring(
            title,
            prompt,
            show="•",
            parent=self.root,
        )
        if pin is not None and self.operator_auth.verify(pin):
            self._operator_unlocked = True
            self._update_operator_ui()
            success_msg = (
                "Attendance administration unlocked"
                if is_attendance_admin
                else "Operator workspace unlocked"
            )
            self.shell.show_toast(success_msg, tone="success")
            return True
        if pin is not None:
            err_msg = (
                "Incorrect administrator PIN"
                if is_attendance_admin
                else "Incorrect operator PIN"
            )
            self.shell.show_toast(err_msg, tone="danger")
        return False

    def lock_operator(self) -> None:
        if not self.operator_auth.enabled:
            self.shell.show_toast(
                "Operator PIN is not configured; supervised mode remains active.",
                tone="warning",
            )
            return
        self._stop_active_session(wait=True)
        self._operator_unlocked = False
        self._update_operator_ui()
        self._active_page = "verify"
        self.shell.select("verify")
        self.view.tkraise()
        toast = (
            "Attendance admin locked"
            if self.operator_auth.role == "attendance_admin"
            else "Operator workspace locked"
        )
        self.shell.show_toast(toast, tone="neutral")

    def toggle_presentation_mode(self) -> None:
        self._presentation_mode = not self._presentation_mode
        self.root.attributes("-fullscreen", self._presentation_mode)
        self.shell.show_toast(
            "Presentation mode enabled. Press Esc to exit."
            if self._presentation_mode
            else "Presentation mode disabled",
            tone="neutral",
        )

    def _leave_presentation_mode(self, _event: tk.Event | None = None) -> None:
        if self._presentation_mode:
            self._presentation_mode = False
            self.root.attributes("-fullscreen", False)
            self.shell.show_toast("Presentation mode disabled", tone="neutral")

    def show_page(self, name: str) -> None:
        page = self._pages.get(name)
        if page is None:
            return
        if not self.operator_auth.permits(name):
            self.shell.show_toast(
                "This administrator can manage attendance only.", tone="warning"
            )
            return
        if name in {"enroll", "users", "history", "settings"} and not self._ensure_operator():
            return
        if getattr(self, "_auto_reset_timer", None) is not None:
            try:
                self.root.after_cancel(self._auto_reset_timer)
            except (tk.TclError, AttributeError):
                pass
            self._auto_reset_timer = None
        if (
            self.operator_auth.enabled
            and self._active_page in {"enroll", "users", "history", "settings"}
            and name == "verify"
        ):
            self._operator_unlocked = False
            self._update_operator_ui()
        if name != self._active_page:
            self._stop_active_session(wait=True)
            self._clear_verification_state()
        self._active_page = name
        self.shell.select(name)
        page.tkraise()
        if name == "users":
            self.refresh_users()
        elif name == "history":
            self.refresh_history()
        elif name == "enroll":
            self.enrollment_view.set_camera_controls(
                start=self._bundle is not None, stop=False, capture=False
            )

    def check_system(self) -> None:
        self._stop_active_session(wait=True)
        if self._bundle is not None:
            self._bundle.verifier.end()
        self._readiness_generation += 1
        generation = self._readiness_generation
        self._bundle = None
        self._clear_verification_state()
        self.view.clear_preview("Preparing FaceGuard…")
        self.view.set_status(
            "PREPARING",
            "Please wait while FaceGuard gets ready.",
            "neutral",
        )
        self.view.set_controls(check=False, start=False, stop=False)
        self.enrollment_view.set_camera_controls(start=False, stop=False, capture=False)

        def load() -> None:
            try:
                bundle = RuntimeFactory.load(self.config_path, project_root=self.project_root)
                update = (generation, True, bundle)
            except Exception as exc:
                update = (generation, False, exc)
            try:
                self._readiness_updates.put_nowait(update)
            except queue.Full:
                try:
                    self._readiness_updates.get_nowait()
                except queue.Empty:
                    pass
                self._readiness_updates.put_nowait(update)

        threading.Thread(target=load, name="faceguard-app-readiness", daemon=True).start()

    def _accept_readiness(self, success: bool, payload: Any) -> None:
        if success:
            self._bundle = payload
            self.exit_code = 0
            report = payload.readiness
            providers = ", ".join(report["providers"])
            self.view.set_status("READY", "Press Start camera for the next person.", "success")
            self.view.reset_metrics(
                pad_required=payload.config.pipeline_policy.pad_min_observations
            )
            self.view.set_controls(check=True, start=True, stop=False)
            self.enrollment_view.set_camera_controls(start=True, stop=False, capture=False)
            self.settings_view.update_info(
                provider=providers,
                gallery_identities=report["gallery_identities"],
                gallery_vectors=report["gallery_vectors"],
            )
            if self._active_page == "users":
                self.refresh_users()
            return

        error = payload
        self.exit_code = 2
        self.view.set_status("SYSTEM BLOCKED", f"{type(error).__name__}: {error}", "danger")
        self.view.set_controls(check=True, start=False, stop=False)
        self.enrollment_view.set_camera_controls(start=False, stop=False, capture=False)

    def _start_shared_camera(self, mode: str) -> None:
        if self._bundle is None:
            return
        self._stop_active_session(wait=True)
        self._session = VerificationSession(
            verifier=self._bundle.verifier,
            config=self._bundle.config,
            options=SessionOptions(camera_index=self.camera_index),
        )
        self._session_mode = mode
        self._latest_snapshot = None
        self._latest_verified = None
        self._checkin_inflight = False
        self._recorded_checkin_key = None
        self._verification_finishing = False
        try:
            self._session.start()
        except Exception as exc:
            self.exit_code = 2
            if mode == "verification":
                self.view.set_status("CAMERA ERROR", f"{type(exc).__name__}: {exc}", "danger")
                self.view.set_controls(check=True, start=True, stop=False)
            else:
                self.enrollment_view.set_samples(0, guidance=f"Camera error: {exc}")
                self.enrollment_view.set_camera_controls(start=True, stop=False, capture=False)

    def start_camera(self) -> None:
        if self._bundle is None:
            return
        if getattr(self, "_auto_reset_timer", None) is not None:
            try:
                self.root.after_cancel(self._auto_reset_timer)
            except (tk.TclError, AttributeError):
                pass
            self._auto_reset_timer = None
        self.view.reset_metrics(
            pad_required=self._bundle.config.pipeline_policy.pad_min_observations
        )
        self.view.clear_preview("Opening authorized camera 0…")
        self.view.set_status(
            "STARTING CAMERA",
            "Camera access is local and frames will not be recorded.",
            "neutral",
        )
        self.view.set_controls(check=False, start=False, stop=True)
        self._start_shared_camera("verification")

    def start_enrollment_camera(self) -> None:
        if not self.enrollment_view.consent_var.get() or not self.enrollment_view.identity_label:
            messagebox.showwarning(
                "Enrollment details required",
                "Enter a non-sensitive identity label and record consent before opening the camera.",
                parent=self.root,
            )
            return
        self.enrollment_view.clear_preview("Opening authorized camera 0…")
        self.enrollment_view.set_samples(
            len(self._enrollment_embeddings),
            guidance="Keep one face centered, well lit and steady.",
            duplicate_ids=self._duplicate_ids,
        )
        self.enrollment_view.set_camera_controls(start=False, stop=True, capture=True)
        self._start_shared_camera("enrollment")

    def _clear_verification_state(self) -> None:
        self._latest_snapshot = None
        self._latest_verified = None

    def _stop_active_session(self, *, wait: bool) -> None:
        session = self._session
        if session is not None:
            session.stop(wait=wait, timeout=2.0)
        if wait or session is None or not session.threads_alive:
            self._session = None
            self._session_mode = None
        self._clear_verification_state()

    def stop_camera(self) -> None:
        if getattr(self, "_auto_reset_timer", None) is not None:
            try:
                self.root.after_cancel(self._auto_reset_timer)
            except (tk.TclError, AttributeError):
                pass
            self._auto_reset_timer = None
        self._verification_finishing = False
        mode = self._session_mode
        if mode == "enrollment":
            self.enrollment_view.clear_preview("Camera preview is off")
            self.enrollment_view.set_camera_controls(
                start=self._bundle is not None, stop=False, capture=False
            )
        else:
            self.view.clear_preview("Camera preview is off")
            self.view.set_status(
                "READY", "Press Start camera for the next person.", "success"
            )
            self.view.set_source_active(False)
            self.view.set_controls(check=False, start=self._bundle is not None, stop=False)
        self._stop_active_session(wait=False)

    def capture_enrollment_sample(self) -> None:
        if (
            self._sample_inflight
            or self._session is None
            or self._session_mode != "enrollment"
            or self._latest_snapshot is None
        ):
            return
        frame = self._latest_snapshot.frame_bgr.copy()
        session = self._session
        self._sample_inflight = True
        self.enrollment_view.set_camera_controls(start=False, stop=True, capture=False)
        self.enrollment_view.set_busy(True, "Checking face count, lighting, focus and face size…")
        self._run_task("enrollment_sample", lambda: session.enrollment_observation(frame))

    def clear_enrollment(self) -> None:
        self._stop_active_session(wait=True)
        self._enrollment_embeddings.clear()
        self._duplicate_ids = ()
        self._sample_inflight = False
        self.enrollment_view.reset()
        self.enrollment_view.set_busy(False, "Record consent, then capture five quality samples.")
        self.enrollment_view.set_camera_controls(
            start=self._bundle is not None, stop=False, capture=False
        )

    def confirm_enrollment(self) -> None:
        label = self.enrollment_view.identity_label
        if len(self._enrollment_embeddings) < self.enrollment_view.SAMPLE_TARGET:
            return
        if not self.enrollment_view.consent_var.get():
            messagebox.showwarning("Consent required", "Recorded consent is required.", parent=self.root)
            return
        if self._duplicate_ids and not self.enrollment_view.review_var.get():
            messagebox.showwarning(
                "Duplicate review required",
                "Review and acknowledge the possible duplicate identity warning.",
                parent=self.root,
            )
            return
        duplicate_note = (
            f"\n\nPossible duplicates reviewed: {', '.join(self._duplicate_ids)}"
            if self._duplicate_ids
            else ""
        )
        if not messagebox.askyesno(
            "Confirm enrollment",
            f"Enroll '{label}' with {len(self._enrollment_embeddings)} vectors?"
            f"\n\nConsent recorded: Yes\nRaw images retained: No{duplicate_note}",
            parent=self.root,
        ):
            return
        self._stop_active_session(wait=True)
        matrix = np.stack(self._enrollment_embeddings).astype(np.float32, copy=True)
        duplicates = self._duplicate_ids
        reviewed = self.enrollment_view.review_var.get()
        self.enrollment_view.set_busy(True, "Saving the protected gallery and verifying its integrity…")
        self.enrollment_view.set_camera_controls(start=False, stop=False, capture=False)
        self._run_task(
            "gallery_enroll",
            lambda: enroll_identity(
                self.gallery_path,
                label,
                matrix,
                consent_recorded=True,
                duplicate_candidates_reviewed=reviewed,
                duplicate_identity_ids=duplicates,
                runtime_config_path=self.config_path,
            ),
        )

    def record_attendance(self) -> None:
        verified = self._latest_verified
        if verified is None or self._checkin_inflight:
            return
        person_id, model_version, track_id = verified
        self._checkin_inflight = True
        self._recorded_checkin_key = (person_id, track_id)
        self.view.set_attendance_status("Marking today's attendance…", "neutral")
        self._run_task(
            "attendance_record",
            lambda: self.history_store.record_attendance(
                person_id,
                app_version=__version__,
                model_version=model_version,
            ),
        )

    def _auto_reset_after_attendance(self) -> None:
        self._auto_reset_timer = None
        if self._session is None and self._active_page == "verify":
            self.view.clear_preview("Camera preview is off")
            self.view.set_status(
                "READY",
                "Press Start camera for the next person.",
                "success",
            )
            self.view.set_source_active(False)
            self.view.set_controls(check=False, start=self._bundle is not None, stop=False)

    def _finish_verified_session(self, session: VerificationSession) -> None:
        """Freeze the accepted frame, save attendance, and release the camera."""
        self._verification_finishing = True
        self.record_attendance()
        self.view.set_controls(check=False, start=False, stop=True)
        session.stop(wait=False)

    def refresh_users(self) -> None:
        path = self.gallery_path
        self.users_view.set_status("Loading protected gallery metadata…")
        self._run_task("users_list", lambda: list_identities(path))

    def inspect_selected_user(self) -> None:
        person_id = self.users_view.selected_person_id
        if person_id is None:
            messagebox.showinfo("Select an identity", "Select one enrolled identity first.", parent=self.root)
            return
        self.users_view.set_status("Loading enrollment metadata…")
        self._run_task("user_inspect", lambda: inspect_identity(self.gallery_path, person_id))

    def delete_selected_user(self) -> None:
        person_id = self.users_view.selected_person_id
        if person_id is None:
            messagebox.showinfo("Select an identity", "Select one enrolled identity first.", parent=self.root)
            return
        display = self.identity_labels.get(person_id, person_id)
        if not messagebox.askyesno(
            "Delete enrolled identity",
            f"Permanently remove '{display}' ({person_id}) from the active gallery?\n\n"
            "This removes all stored embedding vectors for this identity and cannot be undone from the app.",
            icon="warning",
            parent=self.root,
        ):
            return
        self._stop_active_session(wait=True)
        self.users_view.set_status(f"Deleting {person_id} and verifying the updated gallery…")
        self._run_task(
            "user_delete",
            lambda: delete_identity(
                self.gallery_path,
                person_id,
                confirmed=True,
                runtime_config_path=self.config_path,
            ),
        )

    def refresh_history(self) -> None:
        self.history_view.set_status("Loading local attendance…")
        self._run_task("history_list", lambda: self.history_store.recent(limit=500))

    def export_history(self) -> None:
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export permitted attendance records",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"),),
        )
        if not target:
            return
        self.history_view.set_status("Exporting CSV…")
        self._run_task("history_export", lambda: self.history_store.export_csv(Path(target)))

    def clear_history(self) -> None:
        if not messagebox.askyesno(
            "Delete all attendance history",
            "Delete every local attendance record? This cannot be undone.",
            icon="warning",
            parent=self.root,
        ):
            return
        self.history_view.set_status("Deleting local history…")
        self._run_task("history_clear", self.history_store.clear)

    def delete_selected_attendance(self) -> None:
        event_id = self.history_view.selected_event_id
        if event_id is None:
            messagebox.showinfo(
                "Select attendance",
                "Select an attendance record, not a month or day heading.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Delete selected attendance",
            f"Delete attendance record {event_id}? This cannot be undone.",
            icon="warning",
            parent=self.root,
        ):
            return
        self.history_view.set_status("Deleting selected attendance…")
        self._run_task(
            "history_delete", lambda: self.history_store.delete_event(event_id)
        )

    def reset_demo_session(self) -> None:
        if not self.operator_auth.permits("settings") or not self._ensure_operator():
            return
        if not messagebox.askyesno(
            "Reset demo session",
            "Stop the camera, clear unfinished enrollment samples, and delete local "
            "attendance history?\n\nEnrolled identities will not be changed.",
            icon="warning",
            parent=self.root,
        ):
            return
        self._stop_active_session(wait=True)
        self._enrollment_embeddings.clear()
        self._duplicate_ids = ()
        self._sample_inflight = False
        self.enrollment_view.reset()
        self.view.clear_preview("Camera preview is off")
        self._run_task("demo_reset", self.history_store.clear)

    def _handle_task_update(self, name: str, success: bool, payload: Any) -> None:
        if name == "enrollment_sample":
            self._sample_inflight = False
            self.enrollment_view.set_busy(False, "Enrollment sample analysis complete.")
            if not success:
                self.enrollment_view.set_samples(
                    len(self._enrollment_embeddings), guidance=f"Sample failed safely: {payload}"
                )
            elif payload.reason is not None:
                guidance = REASON_GUIDANCE.get(payload.reason, payload.reason.replace("_", " ").title())
                self.enrollment_view.set_samples(len(self._enrollment_embeddings), guidance=guidance)
            else:
                self._enrollment_embeddings.append(payload.embedding)
                assert self._bundle is not None
                self._duplicate_ids = duplicate_candidates(
                    self._bundle.verifier.gallery,
                    np.stack(self._enrollment_embeddings),
                    threshold=self._bundle.config.pipeline_policy.recognition_threshold,
                    min_margin=self._bundle.config.pipeline_policy.recognition_min_margin,
                )
                count = len(self._enrollment_embeddings)
                self.enrollment_view.set_samples(
                    count,
                    guidance="Sample accepted. Change pose slightly and capture the next sample.",
                    duplicate_ids=self._duplicate_ids,
                )
                if count >= self.enrollment_view.SAMPLE_TARGET:
                    self._stop_active_session(wait=False)
                    self.enrollment_view.clear_preview("Capture complete. Review and confirm.")
            can_capture = (
                self._session is not None
                and self._session_mode == "enrollment"
                and len(self._enrollment_embeddings) < self.enrollment_view.SAMPLE_TARGET
            )
            self.enrollment_view.set_camera_controls(
                start=self._bundle is not None and self._session is None,
                stop=self._session is not None,
                capture=can_capture,
            )
            return

        if not success:
            if name == "attendance_record":
                self._checkin_inflight = False
                self._recorded_checkin_key = None
                self.view.set_status(
                    "ATTENDANCE NOT SAVED",
                    "Verification finished, but attendance could not be saved. Start again to retry.",
                    "danger",
                )
                self.view.set_controls(
                    check=False,
                    start=self._session is None and self._bundle is not None,
                    stop=False,
                )
            messagebox.showerror("Operation failed", f"{type(payload).__name__}: {payload}", parent=self.root)
            if name.startswith("history"):
                self.history_view.set_status(str(payload), error=True)
            elif name.startswith("user"):
                self.users_view.set_status(str(payload))
            elif name == "gallery_enroll":
                self.enrollment_view.set_busy(False, f"Enrollment failed safely: {payload}")
                self.enrollment_view.set_samples(
                    len(self._enrollment_embeddings),
                    guidance=f"Enrollment failed safely: {payload}",
                    duplicate_ids=self._duplicate_ids,
                )
            return

        if name == "gallery_enroll":
            enrolled_id = payload["person_id"]
            self._enrollment_embeddings.clear()
            self._duplicate_ids = ()
            self.enrollment_view.mark_complete()
            self.enrollment_view.set_busy(False, f"Enrollment saved for {enrolled_id}.")
            self.shell.show_toast(
                f"Enrollment saved for {enrolled_id}. Gallery integrity verified.",
                tone="success",
            )
            self.check_system()
            self.show_page("users")
        elif name == "attendance_record":
            self._checkin_inflight = False
            if payload is None:
                self.view.set_status(
                    "ALREADY MARKED",
                    "This person's attendance was already recorded today. The camera has stopped.",
                    "success",
                )
                self.shell.show_toast("Attendance was already marked today", tone="neutral")
            else:
                display_name = self.identity_labels.get(
                    payload.identity_reference, payload.identity_reference
                )
                self.view.set_status(
                    "ATTENDANCE MARKED",
                    f"{display_name} was verified and marked present for {payload.local_date}.",
                    "success",
                )
                self.shell.show_toast("Attendance marked successfully", tone="success")
            self.view.set_source_active(False)
            self.view.set_controls(
                check=False,
                start=self._session is None and self._bundle is not None,
                stop=True,
            )
            if self._auto_reset_timer is not None:
                try:
                    self.root.after_cancel(self._auto_reset_timer)
                except tk.TclError:
                    pass
            self._auto_reset_timer = self.root.after(4000, self._auto_reset_after_attendance)
        elif name == "users_list":
            self.users_view.set_rows(payload["identities"], self.identity_labels)
            self.users_view.set_status(
                f"{payload['identities_count']} identities, {payload['vectors_count']} vectors"
            )
        elif name == "user_inspect":
            self.users_view.show_details(payload)
        elif name == "user_delete":
            self.users_view.set_status(
                f"Deleted {payload['person_id']}. {payload['remaining_identities']} identities remain."
            )
            self.check_system()
            self.refresh_users()
            self.shell.show_toast(
                f"{payload['person_id']} removed from the active gallery", tone="success"
            )
        elif name == "history_list":
            self.history_view.set_events(payload, self.identity_labels)
        elif name == "history_export":
            self.history_view.set_status(f"Exported {payload} permitted record(s)")
            self.shell.show_toast(f"Exported {payload} attendance record(s)", tone="success")
        elif name == "history_clear":
            self.history_view.set_status(f"Deleted {payload} record(s)")
            self.refresh_history()
            self.shell.show_toast("Attendance history deleted", tone="success")
        elif name == "history_delete":
            self.history_view.set_status(
                "Selected attendance deleted" if payload else "Attendance was already removed"
            )
            self.refresh_history()
            self.shell.show_toast("Selected attendance removed", tone="success")
        elif name == "demo_reset":
            self.history_view.set_events((), self.identity_labels)
            self.shell.show_toast(
                f"Demo session reset. {payload} attendance record(s) deleted.",
                tone="success",
            )

    def _poll(self) -> None:
        if self._closed:
            return
        while True:
            try:
                generation, success, payload = self._readiness_updates.get_nowait()
            except queue.Empty:
                break
            if generation == self._readiness_generation:
                self._accept_readiness(success, payload)
        while True:
            try:
                name, success, payload = self._task_updates.get_nowait()
            except queue.Empty:
                break
            self._handle_task_update(name, success, payload)

        session = self._session
        if session is not None:
            snapshot = session.latest_snapshot()
            if snapshot is not None and self._bundle is not None:
                self._latest_snapshot = snapshot
                if self._session_mode == "verification" and not self._verification_finishing:
                    self.view.render_snapshot(
                        snapshot,
                        pad_required=self._bundle.config.pipeline_policy.pad_min_observations,
                    )
                    result = snapshot.result
                    if result.status == "verified" and result.person_id:
                        self._latest_verified = (
                            result.person_id,
                            result.model_version,
                            result.track_id,
                        )
                        key = (result.person_id, result.track_id)
                        if not self._checkin_inflight and self._recorded_checkin_key != key:
                            self._finish_verified_session(session)
                    else:
                        self._latest_verified = None
                elif self._session_mode == "enrollment":
                    self.enrollment_view.render_frame(snapshot.frame_bgr)

            status = session.status
            if status == SessionStatus.RECONNECTING:
                self._clear_verification_state()
                if self._session_mode == "verification":
                    self.view.set_status(
                        "CAMERA UNAVAILABLE",
                        "Camera input was interrupted. Verification state has been cleared.",
                        "warning",
                    )
                    self.view.set_source_active(False)
                else:
                    self.enrollment_view.clear_preview("Camera unavailable. Reconnecting safely.")
                    self.enrollment_view.set_camera_controls(start=False, stop=True, capture=False)
            elif status == SessionStatus.FAILED and not session.threads_alive:
                error = session.error or "The camera session failed safely."
                mode = self._session_mode
                self._session = None
                self._session_mode = None
                self._clear_verification_state()
                if mode == "verification":
                    self.view.clear_preview("Camera session ended safely")
                    self.view.set_status("CAMERA UNAVAILABLE", error, "danger")
                    self.view.set_controls(check=True, start=self._bundle is not None, stop=False)
                else:
                    self.enrollment_view.clear_preview("Camera session ended safely")
                    self.enrollment_view.set_samples(
                        len(self._enrollment_embeddings), guidance=error, duplicate_ids=self._duplicate_ids
                    )
                    self.enrollment_view.set_camera_controls(
                        start=self._bundle is not None, stop=False, capture=False
                    )
            elif status == SessionStatus.STOPPED and not session.threads_alive:
                mode = self._session_mode
                verification_finishing = self._verification_finishing
                self._session = None
                self._session_mode = None
                self._clear_verification_state()
                if mode == "verification":
                    if verification_finishing:
                        self._verification_finishing = False
                        self.view.set_source_active(False)
                        if self._checkin_inflight:
                            self.view.set_status(
                                "VERIFIED",
                                "Saving attendance. Please wait a moment.",
                                "success",
                            )
                        self.view.set_controls(
                            check=False,
                            start=not self._checkin_inflight and self._bundle is not None,
                            stop=True,
                        )
                    else:
                        self.view.clear_preview("Camera preview is off")
                        self.view.set_status(
                            "READY",
                            "Press Start camera for the next person.",
                            "success",
                        )
                        self.view.set_controls(
                            check=False, start=self._bundle is not None, stop=False
                        )
                elif len(self._enrollment_embeddings) < self.enrollment_view.SAMPLE_TARGET:
                    self.enrollment_view.clear_preview()
                    self.enrollment_view.set_camera_controls(
                        start=self._bundle is not None, stop=False, capture=False
                    )

        self._poll_id = self.root.after(self.POLL_MS, self._poll)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_auto_reset_timer", None) is not None:
            try:
                self.root.after_cancel(self._auto_reset_timer)
            except (tk.TclError, AttributeError):
                pass
            self._auto_reset_timer = None
        try:
            self.root.after_cancel(self._poll_id)
        except tk.TclError:
            pass
        self._stop_active_session(wait=True)
        if self._bundle is not None:
            self._bundle.verifier.end()
        self._enrollment_embeddings.clear()
        self.root.destroy()

    def run(self) -> int:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.exit_code = 130
            self.close()
        return self.exit_code
