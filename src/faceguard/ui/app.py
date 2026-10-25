"""Application controller for the FaceGuard single-window desktop interface."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from typing import Any

from ..bootstrap import RuntimeBundle, RuntimeFactory
from ..identity_labels import load_identity_labels
from ..session import SessionOptions, SessionStatus, VerificationSession
from .live_view import LiveView
from .theme import configure_root


class DesktopApp:
    """Coordinate readiness, camera lifecycle and UI updates."""

    POLL_MS = 40

    def __init__(
        self,
        root: tk.Tk,
        *,
        config_path: Path,
        project_root: Path,
        camera_index: int = 0,
        identity_labels_path: Path | None = None,
    ) -> None:
        if camera_index != 0:
            raise ValueError("Only authorized camera 0 is supported")
        self.root = root
        self.config_path = config_path
        self.project_root = project_root
        self.camera_index = camera_index
        self.identity_labels = load_identity_labels(identity_labels_path)
        self.exit_code = 0
        self._closed = False
        self._bundle: RuntimeBundle | None = None
        self._session: VerificationSession | None = None
        self._readiness_generation = 0
        self._readiness_updates: queue.Queue[tuple[int, bool, Any]] = queue.Queue(maxsize=2)

        root.title("FaceGuard AI · Local Vision Verification")
        root.geometry("1180x790")
        root.minsize(980, 680)
        configure_root(root)
        self.view = LiveView(
            root,
            on_check=self.check_system,
            on_start=self.start_camera,
            on_stop=self.stop_camera,
            identity_labels=self.identity_labels,
        )
        self.view.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(120, self.check_system)
        self._poll_id = root.after(self.POLL_MS, self._poll)

    def check_system(self) -> None:
        if self._session is not None and self._session.is_running:
            return
        if self._session is not None:
            self._session.close()
            self._session = None
        elif self._bundle is not None:
            self._bundle.verifier.end()
        self._readiness_generation += 1
        generation = self._readiness_generation
        self._bundle = None
        self.view.clear_preview("Preparing the local vision runtime…")
        self.view.set_status(
            "CHECKING SYSTEM",
            "Validating model hashes, CPU providers and the protected gallery.",
            "neutral",
        )
        self.view.set_readiness("Runtime validation in progress")
        self.view.set_controls(check=False, start=False, stop=False)

        def load() -> None:
            try:
                bundle = RuntimeFactory.load(
                    self.config_path,
                    project_root=self.project_root,
                )
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

        threading.Thread(
            target=load,
            name="faceguard-app-readiness",
            daemon=True,
        ).start()

    def _accept_readiness(self, success: bool, payload: Any) -> None:
        if success:
            self._bundle = payload
            self.exit_code = 0
            report = payload.readiness
            providers = ", ".join(report["providers"])
            self.view.set_status(
                "SYSTEM READY",
                "All configured artifacts passed integrity and provider checks.",
                "success",
            )
            self.view.set_readiness(
                f"{report['release_id']}  ·  {providers}\n"
                f"{report['gallery_identities']} identities  ·  "
                f"{report['gallery_vectors']} gallery vectors"
            )
            self.view.reset_metrics(
                pad_required=payload.config.pipeline_policy.pad_min_observations
            )
            self.view.set_controls(check=True, start=True, stop=False)
            return

        error = payload
        self.exit_code = 2
        self.view.set_status(
            "SYSTEM BLOCKED",
            f"{type(error).__name__}: {error}",
            "danger",
        )
        self.view.set_readiness(
            "Camera access remains disabled. Correct the local configuration or artifacts, then retry."
        )
        self.view.set_controls(check=True, start=False, stop=False)

    def start_camera(self) -> None:
        if self._bundle is None:
            return
        if self._session is not None:
            if self._session.threads_alive:
                return
            self._session.stop(wait=True)
        self._session = VerificationSession(
            verifier=self._bundle.verifier,
            config=self._bundle.config,
            options=SessionOptions(camera_index=self.camera_index),
            event_sink=self._emit_event,
        )
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
        self.exit_code = 0
        try:
            self._session.start()
        except Exception as exc:
            self.exit_code = 2
            self.view.set_status(
                "CAMERA ERROR",
                f"{type(exc).__name__}: {exc}",
                "danger",
            )
            self.view.set_controls(check=True, start=True, stop=False)

    @staticmethod
    def _emit_event(event: dict[str, object]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)

    def stop_camera(self) -> None:
        if self._session is None:
            return
        self.view.set_status(
            "STOPPING",
            "Releasing the camera and clearing session state.",
            "neutral",
        )
        self.view.clear_preview("Stopping camera…")
        self.view.set_controls(check=False, start=False, stop=False)
        self._session.stop(wait=False)

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

        session = self._session
        if session is not None:
            snapshot = session.latest_snapshot()
            if snapshot is not None and self._bundle is not None:
                self.view.render_snapshot(
                    snapshot,
                    pad_required=self._bundle.config.pipeline_policy.pad_min_observations,
                )

            status = session.status
            if status == SessionStatus.RECONNECTING:
                self.view.set_status(
                    "RECONNECTING",
                    "Camera input was interrupted. Verification state has been cleared.",
                    "warning",
                )
                self.view.set_source_active(False)
            elif status == SessionStatus.FAILED and not session.threads_alive:
                self.exit_code = 2
                self.view.clear_preview("Camera session ended safely")
                self.view.set_status(
                    "CAMERA ERROR",
                    session.error or "The camera session failed safely.",
                    "danger",
                )
                self.view.set_controls(check=True, start=self._bundle is not None, stop=False)
            elif status == SessionStatus.STOPPED and not session.threads_alive:
                self.view.clear_preview("Camera preview is off")
                self.view.set_status(
                    "SYSTEM READY",
                    "Camera stopped. Start a new local verification session when ready.",
                    "success",
                )
                self.view.set_controls(check=True, start=self._bundle is not None, stop=False)

        self._poll_id = self.root.after(self.POLL_MS, self._poll)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.after_cancel(self._poll_id)
        except tk.TclError:
            pass
        if self._session is not None:
            self._session.close()
        elif self._bundle is not None:
            self._bundle.verifier.end()
        self.root.destroy()

    def run(self) -> int:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.exit_code = 130
            self.close()
        return self.exit_code
