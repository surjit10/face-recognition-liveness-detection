"""Professional dark visual system for the local desktop application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


COLORS = {
    "background": "#080D17",
    "surface": "#101827",
    "surface_alt": "#151F31",
    "surface_hover": "#1B2940",
    "border": "#24324A",
    "text": "#F7F9FC",
    "muted": "#8FA0B8",
    "subtle": "#5E708A",
    "accent": "#4F8CFF",
    "accent_hover": "#3978E7",
    "success": "#16A36A",
    "success_soft": "#123C31",
    "warning": "#E6A23C",
    "warning_soft": "#49351A",
    "danger": "#EF5A6F",
    "danger_soft": "#4B202A",
    "neutral": "#7190BA",
    "neutral_soft": "#1C2D45",
    "video": "#050810",
}

FONT_FAMILY = "DejaVu Sans"


def configure_root(root: tk.Misc) -> None:
    root.configure(bg=COLORS["background"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "FaceGuard.Horizontal.TProgressbar",
        troughcolor=COLORS["surface_alt"],
        background=COLORS["accent"],
        bordercolor=COLORS["surface_alt"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
        thickness=8,
    )


def button(
    parent: tk.Misc,
    *,
    text: str,
    command,
    primary: bool = False,
    destructive: bool = False,
) -> tk.Button:
    background = COLORS["accent"] if primary else COLORS["surface_alt"]
    active = COLORS["accent_hover"] if primary else COLORS["surface_hover"]
    if destructive:
        background = COLORS["danger_soft"]
        active = COLORS["danger"]
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=(FONT_FAMILY, 10, "bold"),
        fg=COLORS["text"],
        bg=background,
        activeforeground=COLORS["text"],
        activebackground=active,
        disabledforeground=COLORS["subtle"],
        relief="flat",
        bd=0,
        padx=18,
        pady=11,
        cursor="hand2",
        highlightthickness=0,
    )
