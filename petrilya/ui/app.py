"""UI entry point: ``petrilya-ui`` command."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from petrilya.ui.main_window import MainWindow

_STYLE_QSS = Path(__file__).with_name("style.qss")


def configure_app(app: QApplication) -> None:
    """Apply Petrilya branding to a QApplication.

    Shared between the production entry point and the marketing-screenshot
    tool so both render with the same fonts and stylesheet.
    """
    app.setApplicationName("Petrilya")
    app.setApplicationDisplayName("Petrilya")
    app.setOrganizationName("Petrilya")
    app.setStyle("Fusion")

    # Editorial sans for everything. IBM Plex Sans on the website — Qt
    # will fall back to Segoe UI / Helvetica if it's not installed.
    base = QFont("IBM Plex Sans", 10)
    base.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(base)

    if _STYLE_QSS.is_file():
        app.setStyleSheet(_STYLE_QSS.read_text(encoding="utf-8"))


def main() -> None:
    app = QApplication(sys.argv)
    configure_app(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
