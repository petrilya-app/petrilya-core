"""UI entry point: ``petrilya-ui`` command."""

from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from petrilya.ui.main_window import MainWindow
from petrilya.ui.theme import current_theme, load_qss


def configure_app(app: QApplication, theme: str | None = None) -> str:
    """Apply Petrilya branding (font + stylesheet) to a QApplication.

    Shared between the production entry point and the marketing-screenshot
    tool so both render with the same look.

    Returns the theme name that was applied (useful for callers that need
    to sync UI state, e.g. the theme-toggle button icon).
    """
    app.setApplicationName("Petrilya")
    app.setApplicationDisplayName("Petrilya")
    app.setOrganizationName("Petrilya")
    app.setStyle("Fusion")

    base = QFont("IBM Plex Sans", 10)
    base.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(base)

    theme = theme or current_theme()
    app.setStyleSheet(load_qss(theme))
    return theme


def apply_theme(app: QApplication, theme: str) -> None:
    """Hot-swap the stylesheet to a different theme."""
    app.setStyleSheet(load_qss(theme))


def main() -> None:
    app = QApplication(sys.argv)
    configure_app(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
