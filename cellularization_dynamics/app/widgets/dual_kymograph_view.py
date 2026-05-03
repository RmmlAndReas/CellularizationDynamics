from PyQt6.QtWidgets import QSplitter
from PyQt6.QtCore import Qt

from .threshold_panel import ThresholdPanel
from .front_panel import FrontPanel
from .result_panel import ResultPanel


class DualKymographView(QSplitter):
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setChildrenCollapsible(False)
        self.threshold_panel = ThresholdPanel()
        self.front_panel = FrontPanel()
        self.result_panel = ResultPanel()
        self.addWidget(self.threshold_panel)
        self.addWidget(self.front_panel)
        self.addWidget(self.result_panel)
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 1)
        self.setStretchFactor(2, 1)
        self.setSizes([420, 420, 420])
