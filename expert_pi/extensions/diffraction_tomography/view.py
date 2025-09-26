from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.gui import main_window

from PySide6 import QtCore

from expert_pi.config import get_config
from expert_pi.gui.data_views import histogram, static_image_viewer, view_with_histogram, xyz_plot
from expert_pi.gui.elements import colored_slider
from expert_pi.extensions.diffraction_tomography import acquisition_widget, analysis_widget


class MainView(QtCore.QObject):
    show_signal = QtCore.Signal()
    hide_signal = QtCore.Signal()

    def __init__(self, window: main_window.MainWindow):
        super().__init__()
        self.window = window

        panel_size = get_config().ui.panel_size
        self.acquisition: acquisition_widget.Acquisition = self.window.central_layout.left_toolbars.add_toolbar(
            acquisition_widget.Acquisition(panel_size)
        )
        self.acquisition.setFixedHeight(380)

        self.analysis: analysis_widget.Analysis = self.window.central_layout.left_toolbars.add_toolbar(
            analysis_widget.Analysis(panel_size)
        )

        self.stem_view = static_image_viewer.StaticImageViewer()
        self.stem_view.name = "3ded_stem_view"
        self.diffraction_view = static_image_viewer.StaticImageViewer()
        self.diffraction_view.name = "3ded_diffraction_view"

        self.stem_histogram = histogram.HistogramView("3ded_stem_histogram", ("BF", "HAADF"))
        self.diffraction_histogram = histogram.HistogramView("3ded_diffraction_histogram", ("Camera",))

        self.stem_view_with_histogram = view_with_histogram.ViewWithHistogram(self.stem_view, self.stem_histogram)
        self.diffraction_view_with_histogram = view_with_histogram.ViewWithHistogram(
            self.diffraction_view, self.diffraction_histogram
        )

        self.slider = colored_slider.AnotatedColoredSlider()
        self.slider.name = "3ded_slider"

        self.plot_widget = xyz_plot.XYZPlot()
        self.plot_widget.name = "xyz_tracking_plot"

        self.window.central_layout.central.add_item(self.stem_view_with_histogram.name, self.stem_view_with_histogram)
        self.window.central_layout.central.add_item(
            self.diffraction_view_with_histogram.name, self.diffraction_view_with_histogram
        )
        self.window.central_layout.central.add_item(self.slider.name, self.slider)
        self.window.central_layout.central.add_item(self.plot_widget.name, self.plot_widget)

        self.left_toolbars = [self.acquisition.name, self.window.stem_tools.name, self.analysis.name]

        self.central_acquisition_layout = {
            "vertical": False,
            "position": 0.618,
            "first": self.window.image_view.name,
            "second": {
                "vertical": True,
                "position": 0.618,
                "first": self.window.diffraction_view.name,
                "second": self.plot_widget.name,
            },
        }

        self.central_analysis_layout = {
            "vertical": True,
            "position": 0.95,
            "first": {
                "vertical": False,
                "position": 0.5,
                "first": self.stem_view.name,
                "second": self.diffraction_view.name,
            },
            "second": self.slider.name,
        }

        self.show_analysis = False

    def show(self):
        self.window.central_layout.set_toolbar_item_names(self.left_toolbars, type="left")
        self.window.central_layout.set_toolbar_item_names(None, type="right")

        self.central_analysis_layout["position"] = 1 - 60 / self.window.geometry().height()
        if self.show_analysis:
            self.window.central_layout.set_central_layout(self.central_analysis_layout)
        else:
            self.window.central_layout.set_central_layout(self.central_acquisition_layout)

        self.show_signal.emit()

    # need to be implemented
    def hide(self):
        # optional control at going out from measurement
        self.hide_signal.emit()
