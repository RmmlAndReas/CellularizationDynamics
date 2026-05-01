from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QGuiApplication, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QSizePolicy


class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._original_pixmap = QPixmap()
        self._preview_visible = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No outputs yet. Click Generate Outputs.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.installEventFilter(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self.image_label)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.scroll, 1)

        self.hint_label = QLabel("")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #8a8a8a; padding-top: 2px;")
        layout.addWidget(self.hint_label)

        self.preview_popup = QLabel()
        self.preview_popup.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
        )
        self.preview_popup.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_popup.setStyleSheet(
            "background-color: #1f1f1f; border: 1px solid #808080; padding: 2px;"
        )
        self.preview_popup.hide()

        self.footer_spacer = QWidget(self)
        self.footer_spacer.setMinimumHeight(0)
        self.footer_spacer.setMaximumHeight(0)
        layout.addWidget(self.footer_spacer)

    def show_png(self, png_path: Path):
        if not png_path.exists():
            self._original_pixmap = QPixmap()
            self.image_label.setText("No outputs yet. Click Generate Outputs.")
            self.image_label.setPixmap(QPixmap())
            self.hint_label.setText("")
            self._hide_preview()
            return

        pix = QPixmap(str(png_path))
        if pix.isNull():
            self._original_pixmap = QPixmap()
            self.image_label.setText(f"Could not load image: {png_path}")
            self.image_label.setPixmap(QPixmap())
            self.hint_label.setText("")
            self._hide_preview()
            return

        self._original_pixmap = pix
        self.image_label.setText("")
        self.hint_label.setText("Click image to enlarge (click again to hide)")
        self._render_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_scaled()

    def _render_scaled(self):
        if self._original_pixmap.isNull():
            return
        self.image_label.setPixmap(
            self._original_pixmap.scaled(
                self.scroll.viewport().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def eventFilter(self, watched, event):
        if watched is self.image_label and not self._original_pixmap.isNull():
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._preview_visible:
                    self._hide_preview()
                else:
                    self._show_preview(event.globalPosition().toPoint())
                return True
        return super().eventFilter(watched, event)

    def _show_preview(self, cursor_pos: QPoint):
        if self._original_pixmap.isNull():
            return
        screen = QGuiApplication.screenAt(cursor_pos) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        max_width = max(300, int(available.width() * 0.65))
        max_height = max(250, int(available.height() * 0.65))
        preview_pix = self._original_pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_popup.setPixmap(preview_pix)
        self.preview_popup.resize(preview_pix.size())
        self._move_preview(cursor_pos)
        self.preview_popup.show()
        self._preview_visible = True

    def _move_preview(self, cursor_pos: QPoint):
        if not self._preview_visible and not self.preview_popup.isVisible():
            return
        screen = QGuiApplication.screenAt(cursor_pos) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        offset = QPoint(24, 24)
        target = cursor_pos + offset
        if target.x() + self.preview_popup.width() > available.right():
            target.setX(max(available.left(), cursor_pos.x() - self.preview_popup.width() - 24))
        if target.y() + self.preview_popup.height() > available.bottom():
            target.setY(max(available.top(), cursor_pos.y() - self.preview_popup.height() - 24))
        self.preview_popup.move(target)

    def _hide_preview(self):
        self.preview_popup.hide()
        self._preview_visible = False

    def set_footer_height(self, height: int):
        h = max(0, int(height))
        self.footer_spacer.setMinimumHeight(h)
        self.footer_spacer.setMaximumHeight(h)
