from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea


class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._original_pixmap = QPixmap()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No figure yet. Click Generate Figure.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll)

        self.footer_spacer = QWidget(self)
        self.footer_spacer.setMinimumHeight(0)
        self.footer_spacer.setMaximumHeight(0)
        layout.addWidget(self.footer_spacer)

    def show_png(self, png_path: Path):
        if not png_path.exists():
            self._original_pixmap = QPixmap()
            self.image_label.setText("No figure yet. Click Generate Figure.")
            self.image_label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(png_path))
        if pix.isNull():
            self._original_pixmap = QPixmap()
            self.image_label.setText(f"Could not load image: {png_path}")
            self.image_label.setPixmap(QPixmap())
            return

        self._original_pixmap = pix
        self.image_label.setText("")
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

    def set_footer_height(self, height: int):
        h = max(0, int(height))
        self.footer_spacer.setMinimumHeight(h)
        self.footer_spacer.setMaximumHeight(h)
