from enum import Enum

from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base
from expert_pi.gui.tools import diffraction_generator, ronchigram
from expert_pi.gui.tools.mask_selector.option_widget import OptionWidget as MaskSelectorOptions


class Selector(Enum):
    mask_selector = 0
    distance_tool = 1
    angle_tool = 2
    ronchigram = 3


class Visualisation(Enum):
    histogram = 0
    cross = 1
    circle = 2
    stage_ranges = 3
    axes_orientation = 4
    size_bar = 5
    status_info = 6


class DiffractionTools(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__(
            "DIFFRACTION TOOLS", images_dir + "section_icons/stem_tools.svg", expand_type="right", panel_size=panel_size
        )
        image_type_options = {"camera": ("CAMERA", ""), "fft": ("FFT", "")}
        self.image_type = buttons.ToolbarMultiButton(image_type_options, default_option="camera")

        selector_options = {s.name: ("", images_dir + "tools_icons/" + s.name + ".svg") for s in Selector}
        self.selectors = buttons.ToolbarMultiButton(selector_options, no_select=True)  # click the same to deselect all
        self.selectors.buttons[Selector.mask_selector.name].setToolTip("Select mask for 4DSTEM")
        self.selectors.buttons[Selector.distance_tool.name].setToolTip("Distance measurement")
        self.selectors.buttons[Selector.angle_tool.name].setToolTip("Angle measurement")
        self.selectors.buttons[Selector.ronchigram.name].setToolTip("Ronchigram alignment tool")

        visualisation_options = {v.name: ("", images_dir + "tools_icons/" + v.name + ".svg") for v in Visualisation}
        self.visualisation = buttons.ToolbarMultiButton(
            visualisation_options, multi_select=True
        )  # these buttons can be selected simultanously
        self.visualisation.buttons[Visualisation.histogram.name].setToolTip("Show histogram")
        self.visualisation.buttons[Visualisation.cross.name].setToolTip("Show central cross")
        self.visualisation.buttons[Visualisation.circle.name].setToolTip("Show 0.1nm limi radius")
        self.visualisation.buttons[Visualisation.stage_ranges.name].setToolTip("Show stage alpha/beta ranges")
        self.visualisation.buttons[Visualisation.axes_orientation.name].setToolTip("Show XY axis orientation")
        self.visualisation.buttons[Visualisation.size_bar.name].setToolTip("Show size bar")
        self.visualisation.buttons[Visualisation.status_info.name].setToolTip("Show status information")

        for s in Selector:
            self.selectors.buttons[s.name].setStyleSheet("qproperty-iconSize: 23px")
        for v in Visualisation:
            self.visualisation.buttons[v.name].setStyleSheet("qproperty-iconSize: 22px")

        self.content_layout.addWidget(self.image_type, 0, 0, 1, 1)
        self.content_layout.addWidget(self.selectors, 1, 0, 1, 1)
        self.content_layout.addWidget(self.visualisation, 3, 0, 1, 1)

        self.tool_parameter_widget = QtWidgets.QWidget()
        self.tool_parameter_widget.setLayout(QtWidgets.QVBoxLayout())
        self.tool_parameter_widget.layout().setContentsMargins(1, 1, 1, 1)
        self.tool_parameter_widget.setProperty("class", "toolParameters")
        self.tool_parameter_widget.hide()

        self.content_layout.addWidget(self.tool_parameter_widget, 2, 0, 1, 1)

        self.tools_expansions = {
            Selector.mask_selector.name: MaskSelectorOptions(),
            Selector.ronchigram.name: ronchigram.RonchigramWidget(),
        }

        for key, item in self.tools_expansions.items():
            self.tool_parameter_widget.layout().addWidget(item)
            item.hide()

        self.fixed_height = 133
        self.setFixedHeight(self.fixed_height)

        self.diffraction_generator = diffraction_generator.DiffractionGenerator()
        self.open_diffraction_generator = buttons.ToolbarPushButton("Diffraction generator")
        self.content_layout.addWidget(self.open_diffraction_generator, 4, 0, 1, 1)

        self.selectors.clicked.connect(self.expand_tool)

        self.expand(False)

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
