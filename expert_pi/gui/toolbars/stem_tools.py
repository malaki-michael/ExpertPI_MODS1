from enum import Enum

from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base
from expert_pi.gui.tools import xyz_tracking_controller, xyz_tracking_debugger


class Selector(Enum):
    point_selector = 0
    rectangle_selector = 1
    distance_tool = 2
    angle_tool = 3
    xyz_tracking = 4


class Visualisation(Enum):
    histogram = 0
    cross = 1
    stage_ranges = 2
    axes_orientation = 3
    size_bar = 4
    status_info = 5


class StemTools(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("STEM TOOLS", images_dir + "section_icons/stem_tools.svg", panel_size)

        image_type_options = {"BF": ("BF", ""), "HAADF": ("DF", ""), "EDX": ("EDX", "")}
        self.image_type = buttons.ToolbarMultiButton(image_type_options, default_option="BF")

        selector_options = {s.name: ("", images_dir + "tools_icons/" + s.name + ".svg") for s in Selector}
        self.selectors = buttons.ToolbarMultiButton(selector_options, no_select=True)  # click the same to deselect all
        self.selectors.buttons[Selector.point_selector.name].setToolTip("Point selector, single click")
        self.selectors.buttons[Selector.rectangle_selector.name].setToolTip("Rectangle selector, two clicks wizard")
        self.selectors.buttons[Selector.distance_tool.name].setToolTip("Distance measurement, two clicks wizard")
        self.selectors.buttons[Selector.angle_tool.name].setToolTip("Angle measurement, three clicks wizard")
        self.selectors.buttons[Selector.xyz_tracking.name].setToolTip("Track sample with respect to reference")

        visualisation_options = {v.name: ("", images_dir + "tools_icons/" + v.name + ".svg") for v in Visualisation}
        self.visualisation = buttons.ToolbarMultiButton(
            visualisation_options, multi_select=True
        )  # these buttons can be selected simultanously
        self.visualisation.buttons[Visualisation.histogram.name].setToolTip("Show histogram")
        self.visualisation.buttons[Visualisation.cross.name].setToolTip("Show central cross")
        self.visualisation.buttons[Visualisation.stage_ranges.name].setToolTip("Show stage XY ranges")
        self.visualisation.buttons[Visualisation.axes_orientation.name].setToolTip("Show XY axis orientation")
        self.visualisation.buttons[Visualisation.size_bar.name].setToolTip("Show size bar")
        self.visualisation.buttons[Visualisation.status_info.name].setToolTip("Show status information")

        for s in Selector:
            self.selectors.buttons[s.name].setStyleSheet("qproperty-iconSize: 23px")
        for v in Visualisation:
            self.visualisation.buttons[v.name].setStyleSheet("qproperty-iconSize: 22px")

        # self.visualisation.selected = [0]
        # self.visualisation.buttons[0].setProperty("selected", True)

        self.content_layout.addWidget(self.image_type, 0, 0, 1, 1)
        self.content_layout.addWidget(self.selectors, 1, 0, 1, 1)
        self.content_layout.addWidget(self.visualisation, 3, 0, 1, 1)
        # self.cross.clicked.connect(self.select_cross)

        self.tool_parameter_widget = QtWidgets.QWidget()
        self.tool_parameter_widget.setLayout(QtWidgets.QVBoxLayout())
        self.tool_parameter_widget.layout().setContentsMargins(1, 1, 1, 1)
        self.tool_parameter_widget.setProperty("class", "toolParameters")
        self.tool_parameter_widget.hide()

        self.content_layout.addWidget(self.tool_parameter_widget, 2, 0, 1, 1)

        self.xyz_tracking_widget = xyz_tracking_controller.XYZTrackingWidget()
        self.tracking_debugger = xyz_tracking_debugger.XYZTrackingDebugger()

        self.tools_expansions = {Selector.xyz_tracking.name: self.xyz_tracking_widget}

        for key, item in self.tools_expansions.items():
            self.tool_parameter_widget.layout().addWidget(item)
            item.hide()

        self.fixed_height = 110
        self.setFixedHeight(self.fixed_height)

        self.content_layout.setColumnStretch(0, 1)
        self.content_layout.setColumnStretch(1, 1)

        self.selectors.clicked.connect(self.selectors_clicked)

        self.expand(False)

    def expand(self, value):
        super().expand(value)
        if value:
            self.content_layout.setColumnStretch(1, 1)

        else:
            self.content_layout.setColumnStretch(1, 0)

    def selectors_clicked(self, name):
        if name in self.tools_expansions.keys():
            if self.selectors.selected is None or name not in self.selectors.selected:
                self.expand_tool(None)
            else:
                self.expand_tool(name)

    def sizeHint(self):  # noqa: N802
        size_hint = super().sizeHint()
        if self.tool_parameter_widget.isVisible():
            size_hint.setHeight(self.fixed_height + self.tool_parameter_widget.height())
        else:
            size_hint.setHeight(self.fixed_height)
        return size_hint

    def expand_tool(self, name=None):
        if not self.selectors.selected or name not in self.tools_expansions.keys():
            for expansion_name, item in self.tools_expansions.items():
                if item.isVisible():
                    item.hide()
            self.tool_parameter_widget.hide()
            self.setFixedHeight(self.fixed_height)
        else:
            height = 0
            for expansion_name, item in self.tools_expansions.items():
                if name == expansion_name:
                    item.show()
                    height = item.sizeHint().height()
                else:
                    item.hide()
            self.tool_parameter_widget.setFixedHeight(height + 2)  # 2px border
            self.tool_parameter_widget.show()
            self.setFixedHeight(self.fixed_height + self.tool_parameter_widget.height())
        self.toolbar_manager.parent_resized()
