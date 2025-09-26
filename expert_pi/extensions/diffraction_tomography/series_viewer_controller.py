from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expert_pi.app import app
    from expert_pi.extensions.diffraction_tomography.view import MainView
    from expert_pi.gui.data_views import histogram, static_image_viewer
    from expert_pi.gui.elements import colored_slider

import threading
from functools import partial

import numpy as np
from PySide6 import QtCore

from expert_pi.app.states import states_holder
from expert_pi.gui.style import coloring
from expert_pi.gui.tools import point_selector, rectangle
from expert_pi.measurements.data_formats import diffraction_tomography
from expert_pi.stream_processors import normalizer


class SeriesViewerController(QtCore.QObject):
    def __init__(
        self,
        measurement_view: MainView,
        controller: app.MainApp,
        state: states_holder.StatesHolder,
        image_view: static_image_viewer.StaticImageViewer,
        image_histogram: histogram.HistogramView,
        diffraction_view: static_image_viewer.StaticImageViewer,
        diffraction_histogram: histogram.HistogramView,
        slider: colored_slider.AnotatedColoredSlider,
    ):
        super().__init__()
        self.window = measurement_view.window
        self.measurement_view = measurement_view
        self.controller = controller
        self.state = state
        self.image_view = image_view
        self.image_histogram = image_histogram
        self.diffraction_view = diffraction_view
        self.diffraction_histogram = diffraction_histogram
        self.slider = slider

        self.image_view.tools["point_selector"] = point_selector.PointSelector(self.image_view)
        self.image_view.tools["rectangle"] = rectangle.Rectangle(self.image_view)
        self.image_view.tools["rectangle"].show()

        self.image_view.point_selection_signal.connect(self.point_selected)

        self.image_normalizer = normalizer.Normalizer(
            partial(self.update_image, self.image_view), self.update_image_histogram
        )

        self.diffraction_normalizer = normalizer.Normalizer(
            partial(self.update_image, self.diffraction_view), self.update_diffraction_histogram
        )

        self.tilt_series: diffraction_tomography.DiffractionTiltSeries = None

        self.current_index = 0

        self.auto_index_update = True

        self.stem_channel = "BF"

        self.slider.slider.valueChanged.connect(self.slider_value_changed)

        self.image_histogram.histogram_changed.connect(self.image_histogram_ranges_changed)
        self.diffraction_histogram.histogram_changed.connect(self.diffraction_histogram_ranges_changed)

        self.window.stem_tools.selectors.clicked.connect(self.selectors_clicked)

    def slider_value_changed(self, value):
        self.auto_index_update = value >= self.current_index
        self.set_index(value)

    def set_tilt_series(self, tilt_series):
        self.tilt_series = tilt_series
        self.set_index(0)

        self.measurement_view.slider.set_range(
            self.tilt_series.angles[0] / np.pi * 180,
            self.tilt_series.angles[-1] / np.pi * 180,
            len(self.tilt_series.angles),
        )
        self.measurement_view.slider.slider.set_ranges_colors({(0, len(self.tilt_series.angles)): "#555555"})
        self.measurement_view.slider.update()

    def index_measured(self, index):
        if self.auto_index_update:
            self.set_index(index)
            self.slider.slider.blockSignals(True)
            self.slider.slider.setValue(index)
            self.slider.slider.blockSignals(False)

    def set_index(self, index):
        self.current_index = index
        self.image_normalizer.set_image(self.tilt_series.stem_images[self.stem_channel][index])

        if self.image_view.tools["point_selector"].is_active:
            pos = self.image_view.tools["point_selector"].pos()
            self.point_selected(np.array([pos.x(), pos.y()]))
        else:
            self.diffraction_normalizer.set_image(self.tilt_series.diffractions[index])

        scale = self.image_view.scene().width() / self.tilt_series.parameters["diffractions"]["total_pixels"]
        rect = self.tilt_series.diffraction_selections[index] * scale
        rect_count = self.tilt_series.diffraction_selections[index][2:].astype(np.int64)
        self.image_view.tools["rectangle"].set_rectangle(
            rect[1], rect[0], rect[3], rect[2], rect_count[1], rect_count[0]
        )

    def update_image(self, view, image_8b, frame_index, scan_id):
        view.set_image(image_8b)

    def update_image_histogram(self, histogram_data, frame_index, scan_id):
        return self.update_histogram(self.image_histogram, self.image_normalizer, histogram_data, frame_index, scan_id)

    def update_diffraction_histogram(self, histogram_data, frame_index, scan_id):
        return self.update_histogram(
            self.diffraction_histogram, self.diffraction_normalizer, histogram_data, frame_index, scan_id
        )

    def update_histogram(self, histogram, normalizer, histogram_data, frame_index, scan_id):
        histogram.recalculate_polygons(histogram_data)

        if histogram.auto_range_enabled[histogram.channel]:
            mask = histogram_data > 0
            alpha, beta = histogram.set_range([
                np.argmax(mask) / len(mask),
                (len(mask) - np.argmax(mask[::-1])) / len(mask),
            ])
            normalizer.set_alpha_beta(alpha, beta, histogram.amplify[histogram.channel])

        if threading.current_thread() is threading.main_thread():
            histogram.redraw()
        else:
            histogram.update_signal.emit()

    def diffraction_histogram_ranges_changed(self, channel, alpha, beta):
        self.diffraction_normalizer.set_alpha_beta(
            alpha, beta, self.diffraction_histogram.amplify[self.diffraction_histogram.channel]
        )
        self.set_index(self.current_index)

    def image_histogram_ranges_changed(self, channel, alpha, beta):
        self.image_normalizer.set_alpha_beta(alpha, beta, self.image_histogram.amplify[self.image_histogram.channel])
        self.set_index(self.current_index)

    def point_selected(self, position):
        scale = self.image_view.scene().width() / self.tilt_series.parameters["diffractions"]["total_pixels"]
        rect_4d = self.tilt_series.diffraction_selections[self.current_index].astype(np.int64)
        position_4d = np.round(position / scale - 0.5).astype(np.int64)
        position_4d[0] = np.clip(position_4d[0], rect_4d[1], rect_4d[1] + rect_4d[3] - 1)
        position_4d[1] = np.clip(position_4d[1], rect_4d[0], rect_4d[0] + rect_4d[2] - 1)

        pos = (position_4d + 0.5) * scale
        self.image_view.tools["point_selector"].setPos(pos[0], pos[1])

        self.diffraction_normalizer.set_image(
            self.tilt_series.data5D[self.current_index, position_4d[1] - rect_4d[0], position_4d[0] - rect_4d[1], :, :]
        )

    def selectors_clicked(self, name):
        if self.state.interaction_mode == states_holder.StatesHolder.InteractionMode.measurements:
            if "point_selector" in self.window.stem_tools.selectors.selected:
                self.image_view.tools["point_selector"].show()
            else:
                self.image_view.tools["point_selector"].hide()

    def update_series_slider_colors_percent(self, p, type="done", emit_update=False):
        n = self.slider.slider.maximum()
        if type == "done":
            color = coloring.color_done
        elif type == "part":
            color = coloring.color_acquiring
        else:
            color = coloring.color_todo

        if p >= 100:
            ranges_colors = {(0, n): color}
        else:
            ranges_colors = {(0, p / 100 * n): color, (p / 100 * n, n): coloring.color_todo}

        self.slider.slider.set_ranges_colors(ranges_colors)
        if emit_update:
            self.slider.slider.update_signal.emit()

    def update_series_slider_colors(self, i, acquiring=False, emit_update=False):
        n = self.slider.slider.maximum()
        color_acquiring = coloring.color_acquiring if acquiring else coloring.color_moving

        if i == 0:
            ranges_colors = {(0, 0.5): color_acquiring, (0.5, n): coloring.color_todo}
        elif i == n:
            ranges_colors = {(0, n - 0.5): coloring.color_done, (n - 0.5, n): color_acquiring}
        elif i > n:
            ranges_colors = {(0, n): coloring.color_done}
        else:
            ranges_colors = {
                (0, i - 0.5): coloring.color_done,
                (i - 0.5, i + 0.5): color_acquiring,
                (i + 0.5, n): coloring.color_todo,
            }
        self.slider.slider.set_ranges_colors(ranges_colors)
        if emit_update:
            self.slider.slider.update_signal.emit()
