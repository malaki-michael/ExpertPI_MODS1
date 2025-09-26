import numpy as np
from PySide6 import QtCore

from expert_pi.gui.data_views import base_view, image_item
from expert_pi.gui.tools import (
    angle_tool,
    cross,
    distance_tool,
    histogram_manager,
    mask_selector,
    size_bar,
    status_info,
)
from expert_pi.gui.tools.base import Tool


class DiffractionView(base_view.GraphicsView):
    mask_selector_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.name = "diffraction_view"

        image = np.zeros((2, 2), dtype=np.uint8)
        fov = 10

        self.main_image_item = image_item.ImageItem(image, fov)
        self.add_image_item("main", self.main_image_item)

        self.set_fov(10)
        self.set_center_position([0, 0])

        self.tools = {}

        self.tools = {
            "mask_selector": mask_selector.MaskSelector(self),
            "distance_tool": distance_tool.DistanceTool(self),
            "angle_tool": angle_tool.AngleTool(self),
            "ronchigram": Tool(self),
            # ---
            "histogram": histogram_manager.HistogramManager(self),
            "cross": cross.Cross(self),
            "circle": Tool(self),
            "stage_ranges": Tool(self),
            "axes_orientation": Tool(self),
            "size_bar": size_bar.SizeBar(self, diffraction_units=True),
            "status_info": status_info.StatusInfo(self, is_diffraction=True),
        }

    def redraw(self):
        super().redraw()
        if self.tools["status_info"].is_active:
            self.tools["status_info"].update()
