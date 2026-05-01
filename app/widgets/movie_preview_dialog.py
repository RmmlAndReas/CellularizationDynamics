from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _format_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:d}:{s:02d}"


class MoviePreviewDialog(QDialog):
    """
    Modeless preview for Cellularization_trimmed_delta.mp4 (play, pause, seek).

    Uses OpenCV + QLabel instead of QMediaPlayer: conda-forge ``pyqt6`` does not
    include ``PyQt6.QtMultimedia`` bindings.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Delta movie preview")
        self.setMinimumSize(640, 480)

        self._slider_drag = False
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 1
        self._fps = 10.0
        self._current_frame = 0
        self._playing = False
        self._last_rgb: np.ndarray | None = None

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setStyleSheet("background-color: #1a1a1a;")

        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._time_label = QLabel("0:00 / 0:00")

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c00;")
        self._error_label.setWordWrap(True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time_label)

        root = QVBoxLayout(self)
        root.addWidget(self._video_label, 1)
        root.addLayout(controls)
        root.addWidget(self._error_label)

    def set_source(self, mp4_path: Path) -> None:
        self._error_label.setText("")
        self._release_cap()
        self._last_rgb = None
        path = str(mp4_path.resolve())
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._error_label.setText(f"Could not open video: {path}")
            return

        self._cap = cap
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = max(1, n)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1e-3 else 10.0
        self._timer.setInterval(max(1, int(round(1000.0 / self._fps))))

        self._slider.setMaximum(max(0, self._frame_count - 1))
        self._current_frame = 0
        self._playing = False
        self._play_btn.setText("Play")
        self._set_frame(0)

    def _release_cap(self) -> None:
        self._timer.stop()
        self._playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _toggle_play(self) -> None:
        if self._cap is None:
            return
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("Play")
        else:
            if self._current_frame >= self._frame_count - 1:
                self._set_frame(0)
            self._playing = True
            self._play_btn.setText("Pause")
            self._timer.start()

    def _on_timer_tick(self) -> None:
        if not self._playing or self._cap is None:
            return
        next_f = self._current_frame + 1
        if next_f >= self._frame_count:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("Play")
            return
        self._set_frame(next_f)

    def _set_frame(self, idx: int) -> None:
        if self._cap is None:
            return
        idx = int(np.clip(idx, 0, max(0, self._frame_count - 1)))
        self._current_frame = idx
        if not self._slider_drag:
            self._slider.blockSignals(True)
            self._slider.setValue(idx)
            self._slider.blockSignals(False)
        self._render_current_frame()
        self._update_time_labels()

    def _render_current_frame(self) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_rgb = np.ascontiguousarray(rgb)
        self._apply_pixmap_from_rgb(self._last_rgb)

    def _apply_pixmap_from_rgb(self, rgb: np.ndarray) -> None:
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        scaled = pix.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def _update_time_labels(self) -> None:
        if self._fps <= 0:
            return
        pos_ms = int(self._current_frame / self._fps * 1000)
        dur_ms = int(max(0, self._frame_count - 1) / self._fps * 1000)
        self._time_label.setText(f"{_format_ms(pos_ms)} / {_format_ms(dur_ms)}")

    def _on_slider_pressed(self) -> None:
        self._slider_drag = True

    def _on_slider_released(self) -> None:
        self._slider_drag = False
        self._set_frame(self._slider.value())

    def _on_slider_moved(self, value: int) -> None:
        if not self._slider_drag:
            return
        self._set_frame(value)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_rgb is not None:
            self._apply_pixmap_from_rgb(self._last_rgb)

    def closeEvent(self, event: QCloseEvent):
        self._release_cap()
        self._last_rgb = None
        super().closeEvent(event)
