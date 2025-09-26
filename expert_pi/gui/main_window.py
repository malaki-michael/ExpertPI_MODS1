from PySide6 import QtCore, QtGui, QtWidgets

from expert_pi.config import get_config
from expert_pi.gui import central_layout, navigation_view, survey_view, toolbars
from expert_pi.gui.data_views import diffraction_view, histogram, image_view, spectrum_view, view_with_histogram
from expert_pi.gui.data_views import navigation_view as navigation_data_view
from expert_pi.gui.elements import buttons, combo_box
from expert_pi.gui.style import images_dir
from expert_pi.gui.toolbars import file_saving
from expert_pi.gui.toolbars.diffraction import Diffraction
from expert_pi.gui.toolbars.diffraction_tools import DiffractionTools
from expert_pi.gui.toolbars.stem_adjustments import StemAdjustments


class MenuBar(QtWidgets.QMenuBar):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setStyleSheet("padding:5px 0px;")

        self.file_submenu = self.addMenu("&Microscope")
        self.reconnect_action = QtGui.QAction("Reconnect", self.file_submenu)
        self.file_submenu.addAction(self.reconnect_action)

        self.clear_navigation_action = QtGui.QAction("Clear navigation cache", self.file_submenu)
        self.file_submenu.addAction(self.clear_navigation_action)
        # self.menu_bar.addMenu("&Edit")
        self.view_submenu = self.addMenu("&View")

        combined_action = QtGui.QAction("Combined [num 0]", self.view_submenu)
        # combined_action.triggered.connect(self.main_area.show_combined)
        stem_action = QtGui.QAction("STEM [num 1]", self.view_submenu)
        # stem_action.triggered.connect(self.main_area.show_stem)
        diffraction_action = QtGui.QAction("Diffraction [num 2]", self.view_submenu)
        # diffraction_action.triggered.connect(self.main_area.show_camera)
        spectrum_action = QtGui.QAction("Spectrum [num 3]", self.view_submenu)
        # spectrum_action.triggered.connect(self.main_area.show_spectrum)

        self.view_submenu.addAction(combined_action)
        self.view_submenu.addAction(stem_action)
        self.view_submenu.addAction(diffraction_action)
        self.view_submenu.addAction(spectrum_action)

        # self.menu_bar.addMenu("&Help")


class UpperToolbar(QtWidgets.QToolBar):
    def __init__(self, available_measurements: dict[str, str]):
        super().__init__()

        self.menu = MenuBar()
        self.addWidget(self.menu)

        self.file_saving = file_saving.FileSaving()
        self.addWidget(self.file_saving)
        self.file_saving.layout().setContentsMargins(60, 0, 0, 0)

        self.space = QtWidgets.QWidget()
        self.space.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.addWidget(self.space)

        # visible just under measurement_view:
        self.measurement_file = QtWidgets.QLineEdit("")
        self.measurement_file.setPlaceholderText("In memory")
        self.measurement_file.setEnabled(False)
        self.measurement_file.setFixedWidth(300)

        self.measurement_file_action = self.addWidget(self.measurement_file)

        self.save_measurement_button = buttons.ToolbarPushButton(
            "", icon=images_dir + "tools_icons/save.svg", tooltip="save "
        )
        self.save_as_measurement_button = buttons.ToolbarPushButton(
            "", icon=images_dir + "tools_icons/save_as.svg", tooltip="save as"
        )
        self.load_measurement_button = buttons.ToolbarPushButton(
            "", icon=images_dir + "tools_icons/import.svg", tooltip="load data"
        )
        self.save_measurement_button.setProperty("class", "toolbarButton big")
        self.save_as_measurement_button.setProperty("class", "toolbarButton big")
        self.load_measurement_button.setProperty("class", "toolbarButton big")

        self.save_action = self.addWidget(self.save_measurement_button)
        self.save_as_action = self.addWidget(self.save_as_measurement_button)
        self.load_action = self.addWidget(self.load_measurement_button)

        self.measurement_file_action.setVisible(False)
        self.save_action.setVisible(False)
        self.save_as_action.setVisible(False)
        self.load_action.setVisible(False)

        self.space = QtWidgets.QWidget()
        self.space.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.addWidget(self.space)

        # separate survey widget from the main window
        self.navigation_button = buttons.ToolbarStateButton("Navigation")
        self.navigation_button.setProperty("class", "toolbarButton big")

        self.survey_button = buttons.ToolbarStateButton("Survey", selected=True)
        self.survey_button.setProperty("class", "toolbarButton big")

        self.measurements_options = available_measurements

        self.measurement_button = combo_box.SelectableComboBox(list(self.measurements_options.keys()))

        self.measurement_button.setCurrentIndex(-1)
        self.measurement_button.setPlaceholderText("Measurements")
        self.measurement_button.setMinimumWidth(130)

        self.measurement_button.setProperty("class", "toolbarSelectableComboBox big")

        self.addWidget(self.navigation_button)
        self.addWidget(self.survey_button)
        self.addWidget(self.measurement_button)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.absolute_widgets = {}

        # # TODO: need here to avoid reshow window after first creation of GLWidget
        # QtOpenGLWidgets.QOpenGLWidget(parent=self)
        self.setWindowTitle("Tescan - Expert PI")
        self.setWindowIcon(QtGui.QIcon("expert_pi/gui/style/images/icon.png"))

        self.setStyleSheet(open("expert_pi/gui/style/style.qss", encoding="utf-8").read())

        self.tool_bar = UpperToolbar(get_config().ui.available_measurements)
        self.addToolBar(self.tool_bar)
        self.tool_bar.setMovable(False)

        self.statusBar().showMessage("starting", 1000)

        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setMouseTracking(True)
        self.centralWidget().setMinimumSize(500, 400)

        p_size = get_config().ui.panel_size
        self.central_layout = central_layout.CentralLayout(self.centralWidget(), p_size)
        self.centralWidget().setLayout(self.central_layout)

        # define these first to have them under toolbars if ovelaps:
        self.image_view = image_view.ImageView()
        self.diffraction_view = diffraction_view.DiffractionView()

        self.image_histogram = histogram.HistogramView(
            "image_histogram", ("BF", "HAADF", "EDX", "BF_4DSTEM", "HAADF_4DSTEM", "EDX_4DSTEM")
        )
        self.diffraction_histogram = histogram.HistogramView(
            "diffraction_histogram", ("camera", "fft", "camera_4DSTEM", "fft_4DSTEM")
        )
        self.image_view.tools["histogram"].histogram = self.image_histogram
        self.diffraction_view.tools["histogram"].histogram = self.diffraction_histogram

        self.image_view_with_histogram = view_with_histogram.ViewWithHistogram(self.image_view, self.image_histogram)
        self.diffraction_view_with_histogram = view_with_histogram.ViewWithHistogram(
            self.diffraction_view, self.diffraction_histogram
        )

        self.spectrum_view = spectrum_view.SpectrumView()

        self.central_layout.central.add_item(self.image_view_with_histogram.name, self.image_view_with_histogram)
        self.central_layout.central.add_item(
            self.diffraction_view_with_histogram.name, self.diffraction_view_with_histogram
        )
        self.central_layout.central.add_item(self.spectrum_view.name, self.spectrum_view)

        left_toolbars = self.central_layout.left_toolbars
        right_toolbars = self.central_layout.right_toolbars

        self.optics: toolbars.optics.Optics = left_toolbars.add_toolbar(toolbars.optics.Optics(p_size))
        self.stem_tools: toolbars.stem_tools.StemTools = left_toolbars.add_toolbar(
            toolbars.stem_tools.StemTools(p_size)
        )
        self.scanning: toolbars.scanning.Scanning = left_toolbars.add_toolbar(toolbars.scanning.Scanning(p_size))
        self.detectors: toolbars.detectors.Detectors = left_toolbars.add_toolbar(toolbars.detectors.Detectors(p_size))
        self.stem_4d: toolbars.stem_4d.Stem4D = left_toolbars.add_toolbar(toolbars.stem_4d.Stem4D(p_size))
        self.stem_adjustments: StemAdjustments = left_toolbars.add_toolbar(
            toolbars.stem_adjustments.StemAdjustments(p_size)
        )
        self.camera: toolbars.camera.Camera = right_toolbars.add_toolbar(toolbars.camera.Camera(p_size))
        self.diffraction_tools: DiffractionTools = right_toolbars.add_toolbar(
            toolbars.diffraction_tools.DiffractionTools(p_size)
        )
        self.diffraction: Diffraction = right_toolbars.add_toolbar(toolbars.diffraction.Diffraction(p_size))
        self.precession: toolbars.precession.Precession = right_toolbars.add_toolbar(
            toolbars.precession.Precession(p_size)
        )
        self.xray: toolbars.xray.Xray = right_toolbars.add_toolbar(toolbars.xray.Xray(p_size))

        self.navigation = navigation_data_view.NavigationView()
        self.central_layout.central.add_item("navigation", self.navigation)

        self.navigation_view = navigation_view.NavigationView(self)
        self.survey_view = survey_view.SurveyView(self)
        self.measurement_view = None

        self.measurements = {}

        self.tool_bar.navigation_button.clicked.connect(lambda selected: self.change_view(selected, "navigation"))
        self.tool_bar.survey_button.clicked.connect(lambda selected: self.change_view(selected, "survey"))
        self.tool_bar.measurement_button.clicked.connect(lambda selected: self.change_view(selected, "measurement"))

        self.tool_bar.measurement_button.currentTextChanged.connect(self.change_measurement)

    def change_view(self, selected, type):
        if selected:
            if type == "navigation":
                self.tool_bar.survey_button.set_selected(False)
                self.tool_bar.measurement_button.set_selected(False)
                self.navigation_view.show()

            elif type == "survey":
                self.tool_bar.navigation_button.set_selected(False)
                self.tool_bar.measurement_button.set_selected(False)
                self.survey_view.show()
                self.navigation_view.hide()
            elif type == "measurement":
                self.tool_bar.navigation_button.set_selected(False)
                self.tool_bar.survey_button.set_selected(False)
                self.measurement_view.show()
                self.navigation_view.hide()

    def change_measurement(self, name):
        if self.measurement_view is not None:
            self.measurement_view.hide()
        self.measurement_view = self.measurements[name]
        if self.tool_bar.measurement_button.selected:
            self.measurement_view.show()

    def keyPressEvent(self, event):  # noqa: N802, D102
        if event.key() == QtCore.Qt.Key.Key_R:
            for view in [self.image_view, self.diffraction_view]:
                pos = view.mapFromGlobal(self.cursor().pos())
                size = view.size()
                if pos.x() >= 0 and pos.y() >= 0 and pos.x() < size.width() and pos.y() < size.height():
                    view.set_center_position([0, 0])
                    view.set_fov(view.main_image_item.fov)
                    view.update_tools()
                    event.accept()
                    break

        if event.key() == QtCore.Qt.Key.Key_R or event.key() == QtCore.Qt.Key.Key_A:
            for histogram in [self.image_histogram, self.diffraction_histogram]:
                pos = histogram.mapFromGlobal(self.cursor().pos())
                size = histogram.size()
                if pos.x() >= 0 and pos.y() >= 0 and pos.x() < size.width() and pos.y() < size.height():
                    if event.key() == QtCore.Qt.Key.Key_R:
                        histogram.amplify[histogram.channel] = 1
                        histogram.set_range([0, 1])
                    else:
                        histogram.auto_range()
                    event.accept()
                    break

        elif event.key() == QtCore.Qt.Key.Key_Escape:
            for view in [self.image_view, self.diffraction_view]:
                pos = view.mapFromGlobal(self.cursor().pos())
                size = view.size()
                if pos.x() >= 0 and pos.y() >= 0 and pos.x() < size.width() and pos.y() < size.height():
                    if view == self.image_view and self.stem_tools.selectors.selected is not None:
                        self.stem_tools.selectors.option_clicked(self.stem_tools.selectors.selected)
                    elif view == self.diffraction_view and self.diffraction_tools.selectors.selected is not None:
                        self.diffraction_tools.selectors.option_clicked(self.diffraction_tools.selectors.selected)
        else:
            super().keyPressEvent(event)
