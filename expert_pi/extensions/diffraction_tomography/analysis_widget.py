from PySide6 import QtWidgets

from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import base


class Analysis(base.Toolbar):
    def __init__(self, panel_size: int):
        super().__init__("Analysis", images_dir + "section_icons/processing.svg", panel_size=panel_size)
        self.name = "3ded_analysis"

        layout_options = {"acquisition": ("ACQUISITION", ""), "analysis": ("ANALYSIS", "")}
        self.layout_type = buttons.ToolbarMultiButton(layout_options, default_option="acquisition")
        self.content_layout.addWidget(self.layout_type, 0, 0, 1, 2)

        self.type_processing = combo_box.ToolbarComboBox()
        self.type_processing.addItem("darkfield")
        self.type_processing.addItem("spots")
        self.cut_value = spin_box.SpinBoxWithUnits(40, [-1e32, 1e32], 1, "")

        self.content_layout.addWidget(self.type_processing, 1, 0, 1, 1)
        self.content_layout.addWidget(self.cut_value, 1, 1, 1, 1)

        self.process_button = buttons.ProgressButton("PROCESS", icon=images_dir + "tools_icons/start.svg")

        self.content_layout.addWidget(self.process_button, 2, 0, 1, 2)

        self.content_layout.addWidget(QtWidgets.QLabel(""), 3, 0, 1, 2)

        self.export_tiffs_button = buttons.ToolbarPushButton("Export Tiffs")
        self.content_layout.addWidget(self.export_tiffs_button, 4, 0, 1, 2)
        self.export_videos_button = buttons.ToolbarPushButton("Export Video")
        self.content_layout.addWidget(self.export_videos_button, 5, 0, 1, 2)
        self.export_pets_button = buttons.ToolbarPushButton("Export PETS")
        self.content_layout.addWidget(self.export_pets_button, 6, 0, 1, 2)

        self.expand(False)
