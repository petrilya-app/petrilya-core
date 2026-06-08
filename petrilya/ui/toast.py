"""Non-blocking toast notifications.

Replaces ``QMessageBox`` for ambient feedback (file exported, mask cleared,
batch finished). For modal decisions that need a user response — keep
``QMessageBox.question`` / ``critical``; toasts are fire-and-forget.

Usage::

    from petrilya.ui.toast import toast
    toast(self, "Saved CSV", level="success")
    toast(self, "Mock engine — Cellpose weights pending", level="info")
    toast(self, "Open failed: permission denied", level="error", duration_ms=6000)

Toasts stack from the bottom-right of the parent window.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QTimer,
    Qt,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from petrilya.ui.icons import icon


_LEVEL = {
    "info":    {"color": "#7ab7d0", "icon": "info"},
    "success": {"color": "#6fbfa1", "icon": "check-circle"},
    "error":   {"color": "#e08068", "icon": "alert-triangle"},
}

_MARGIN_X = 20
_MARGIN_Y = 50    # leave room for the status bar
_GAP = 10


class Toast(QFrame):
    """A single toast. Auto-positions, fades in, dismisses after `duration_ms`."""

    _stack: dict[int, list["Toast"]] = {}

    def __init__(
        self,
        parent: QWidget,
        message: str,
        level: str = "info",
        duration_ms: int = 4000,
    ) -> None:
        super().__init__(parent)
        cfg = _LEVEL.get(level, _LEVEL["info"])
        color = cfg["color"]

        self.setObjectName("toastFrame")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Local QSS — global stylesheet doesn't know about this widget.
        self.setStyleSheet(
            f"""
            QFrame#toastFrame {{
                background-color: rgba(22, 27, 36, 240);
                border: 1px solid {color};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
            QFrame#toastFrame QLabel {{
                color: #e8eaef;
                background: transparent;
                font-size: 13px;
            }}
            QFrame#toastFrame QPushButton {{
                background: transparent;
                border: none;
                padding: 2px;
            }}
            QFrame#toastFrame QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.06);
                border-radius: 3px;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 11, 10, 11)
        layout.setSpacing(10)

        ic_label = QLabel()
        ic_label.setPixmap(icon(cfg["icon"], 18, color).pixmap(18, 18))
        layout.addWidget(ic_label, 0, Qt.AlignmentFlag.AlignTop)

        text_label = QLabel(message)
        text_label.setWordWrap(True)
        text_label.setMinimumWidth(220)
        text_label.setMaximumWidth(340)
        layout.addWidget(text_label, 1)

        close_btn = QPushButton()
        close_btn.setIcon(icon("x", 14, "#8a93a4"))
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self._fade_out)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)

        self.adjustSize()

        # Fade-in
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)

        self._register_in_stack()
        self._position()
        self._fade_in()

        if duration_ms > 0:
            QTimer.singleShot(duration_ms, self._fade_out)

    # --------------------------------------------------------- positioning
    def _register_in_stack(self) -> None:
        pid = id(self.parent())
        Toast._stack.setdefault(pid, []).append(self)

    def _unregister(self) -> None:
        pid = id(self.parent())
        lst = Toast._stack.get(pid, [])
        if self in lst:
            lst.remove(self)
        self._reflow_siblings()

    def _reflow_siblings(self) -> None:
        for t in Toast._stack.get(id(self.parent()), []):
            t._position()

    def _position(self) -> None:
        parent = self.parent()
        if not parent:
            return
        pw, ph = parent.width(), parent.height()
        stack = Toast._stack.get(id(parent), [])
        # newest toast at the bottom; stack upward
        try:
            idx = stack.index(self)
        except ValueError:
            idx = 0
        # iterate from bottom (last) upward
        y_offset = _MARGIN_Y
        for t in reversed(stack):
            x = pw - t.width() - _MARGIN_X
            y = ph - t.height() - y_offset
            t.move(x, y)
            y_offset += t.height() + _GAP
        # ensure self also positioned (handled in loop)
        _ = idx

    # ----------------------------------------------------------- animation
    def _fade_in(self) -> None:
        anim = QPropertyAnimation(self._effect, b"opacity", self)
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim
        self.show()
        self.raise_()

    def _fade_out(self) -> None:
        if not self.isVisible():
            return
        anim = QPropertyAnimation(self._effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(self._effect.opacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.finished.connect(self._dismiss)
        anim.start()
        self._anim = anim

    def _dismiss(self) -> None:
        self._unregister()
        self.deleteLater()


def toast(
    parent: QWidget,
    message: str,
    level: str = "info",
    duration_ms: int = 4000,
) -> Toast:
    """Convenience wrapper — show a toast and return its widget."""
    return Toast(parent, message, level, duration_ms)
