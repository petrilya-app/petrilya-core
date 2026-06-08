"""Theme palettes + QSS template substitution.

The single source of truth for colours is the dict in ``PALETTES``.
``style.qss`` is a ``string.Template`` referring to placeholders like
``$bg``, ``$accent`` etc. ``load_qss(name)`` returns the substituted
stylesheet for the requested theme.

Persisted choice lives in ``QSettings`` so the user's preference
carries between launches.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from PySide6.QtCore import QSettings

_QSS = Path(__file__).with_name("style.qss")

THEMES = ("dark", "light")
DEFAULT_THEME = "dark"

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg":           "#0a0d12",
        "bg_soft":      "#11151c",
        "bg_card":      "#161b24",
        "bg_card_alt":  "#141923",
        "bg_hover":     "#1c2230",
        "border":       "#232a39",
        "border_soft":  "#1b2231",
        "border_mid":   "#2a3245",
        "text":         "#e8eaef",
        "text_subtle":  "#c4ccdb",
        "text_dim":     "#8a93a4",
        "text_mute":    "#5e6878",
        "accent":       "#5b9fd1",
        "accent_hover": "#79b3df",
        "accent_press": "#4889bd",
        "accent_text":  "#0b1620",
        "accent_tint":  "rgba(91, 159, 209, 0.10)",
        "accent_tint2": "rgba(91, 159, 209, 0.06)",
        "accent_sel":   "rgba(91, 159, 209, 0.22)",
        "accent_focus": "rgba(91, 159, 209, 0.35)",
        "teal":         "#7ab7d0",
        "input_bg":     "#11151c",
        "scroll":       "#2a3245",
        "scroll_hover": "#3a4358",
        "shadow_alpha": "0.45",
    },
    "light": {
        "bg":           "#f6f4ec",
        "bg_soft":      "#ede9da",
        "bg_card":      "#ffffff",
        "bg_card_alt":  "#faf8f0",
        "bg_hover":     "#f0ede0",
        "border":       "#d8d3c2",
        "border_soft":  "#e4e1d6",
        "border_mid":   "#c8c2ad",
        "text":         "#1d2230",
        "text_subtle":  "#3a4150",
        "text_dim":     "#5b6473",
        "text_mute":    "#8a92a3",
        "accent":       "#3a7fb5",
        "accent_hover": "#4a8ec3",
        "accent_press": "#2e6896",
        "accent_text":  "#ffffff",
        "accent_tint":  "rgba(58, 127, 181, 0.10)",
        "accent_tint2": "rgba(58, 127, 181, 0.06)",
        "accent_sel":   "rgba(58, 127, 181, 0.20)",
        "accent_focus": "rgba(58, 127, 181, 0.32)",
        "teal":         "#2f6a86",
        "input_bg":     "#ffffff",
        "scroll":       "#c8c2ad",
        "scroll_hover": "#a8a288",
        "shadow_alpha": "0.18",
    },
}


def load_qss(theme: str) -> str:
    """Return the QSS for `theme` with all $placeholders substituted.

    Uses ``safe_substitute`` so stray ``$`` characters in QSS comments
    (e.g. the file header that mentions ``$bg``) don't blow up the load.
    Missing palette keys silently stay as ``$key`` — caught quickly in
    visual review rather than crashing at startup.
    """
    if theme not in PALETTES:
        theme = DEFAULT_THEME
    template = Template(_QSS.read_text(encoding="utf-8"))
    return template.safe_substitute(PALETTES[theme])


def _settings() -> QSettings:
    return QSettings("Petrilya", "Petrilya")


def current_theme() -> str:
    """Read the user's saved theme; falls back to dark."""
    name = _settings().value("theme", DEFAULT_THEME, str)
    return name if name in PALETTES else DEFAULT_THEME


def set_current_theme(name: str) -> None:
    """Persist the user's choice. Caller still needs to re-apply QSS."""
    if name in PALETTES:
        _settings().setValue("theme", name)


def toggle_theme() -> str:
    """Flip dark↔light, persist, and return the new theme name."""
    new = "light" if current_theme() == "dark" else "dark"
    set_current_theme(new)
    return new
