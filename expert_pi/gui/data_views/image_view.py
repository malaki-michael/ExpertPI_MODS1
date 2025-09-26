import numpy as np
from PySide6 import QtCore

from expert_pi.app import scan_helper
from expert_pi.gui.data_views import base_view, image_item
from expert_pi.gui.tools import (
    angle_tool,
    cross,
    distance_tool,
    histogram_manager,
    point_selector,
    rectangle_selector,
    reference_multi_image,
    reference_picker,
    size_bar,
    status_info,
)
from expert_pi.gui.tools.base import Tool


class ImageView(base_view.GraphicsView):
    selection_changed_signal = QtCore.Signal()
    point_selection_signal = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.name = "image_view"

        image = np.zeros((2, 2), dtype=np.uint8)
        fov = 10

        self.main_image_item = image_item.ImageItem(image, fov)
        self.add_image_item("main", self.main_image_item)

        self.set_fov(10)

        self.set_center_position([0, 0])

        self.xy_electronic = np.zeros(2)
        self.xy_stage = np.zeros(2)

        self.tools = {
            "point_selector": point_selector.PointSelector(self),
            "rectangle_selector": rectangle_selector.RectangleSelector(self),
            "distance_tool": distance_tool.DistanceTool(self),
            "angle_tool": angle_tool.AngleTool(self),
            "xyz_tracking": reference_multi_image.ReferenceMultiImage(self),
            # ---
            "histogram": histogram_manager.HistogramManager(self),
            "cross": cross.Cross(self),
            "stage_ranges": Tool(self),
            "axes_orientation": Tool(self),
            "size_bar": size_bar.SizeBar(self),
            "status_info": status_info.StatusInfo(self),
            "reference_picker": reference_picker.ReferencePicker(self),
        }

        self.normalizer = None  # need to be inserted by image_view_controller

    def update_transforms(self):
        (
            self.xy_electronic,
            self.xy_stage,
            self.rotation,
            self.transform2x2,
        ) = scan_helper.get_scanning_shift_and_transform()
        # self.axes_orientation.setRotation(-self.rotation / np.pi * 180)  # TODO check definition at non zero alpha beta
        return self.xy_electronic, self.xy_stage, self.rotation, self.transform2x2

    def redraw(self):
        super().redraw()
        if self.tools["status_info"].is_active:
            self.tools["status_info"].update()
