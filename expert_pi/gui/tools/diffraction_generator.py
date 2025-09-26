import os
import types
from os import listdir
from os.path import isfile, join

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLGraphicsItem
from PySide6 import QtCore, QtGui, QtWidgets
from tqdm import tqdm

from expert_pi.gui.console_threads import StoppableThread
from expert_pi.gui.elements import buttons, combo_box, spin_box
from expert_pi.gui.tools import point_group_visualizer
from expert_pi.gui.tools.gl_view import GlViewWidgetOrbit
from expert_pi.measurements.orientation import bragg_fitting


def tqdm_with_callback(callback):
    class tqdm_custom(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs = kwargs.copy()
            kwargs["gui"] = True
            super().__init__(*args, **kwargs)

            if self.disable:
                return

        def clear(self, *_, **__):
            pass

        def display(self, *_, **__):
            callback(self.n, self.total, self.start_t, self._time())

    return tqdm_custom


SIZE = 32


class GLLegend(GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop("glOptions", "additive")
        self.setGLOptions(glopts)
        self.zone_index = [0, 0, 0]
        self.plot_hex = False
        self.angles = [0, 0, 0]

    def update_info(self, zone_index, angles):
        self.zone_index = zone_index
        self.angles = angles

    def paint(self):
        self.setupGLState()

        painter = QtGui.QPainter(self.view())
        self.draw(painter)
        painter.end()

    def draw(self, painter):
        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)

        rect = self.view().rect()
        af = QtCore.Qt.AlignmentFlag
        hex_zone = ""
        if self.plot_hex:
            hex_zone = " hex:" + str(to_hex_indices(self.zone_index))
        lines = [
            "zone: " + str(self.zone_index) + hex_zone,
            f"alpha: {self.angles[0]:6.1f} deg",
            f"beta: {self.angles[1]:6.1f} deg",
            f"gamma: {self.angles[2]:6.1f} deg",
        ]
        info = "\n".join(lines)
        painter.drawText(rect, af.AlignTop | af.AlignLeft, info)


def to_hex_indices(z):
    u = 2 * z[0] - z[1]
    v = 2 * z[1] - z[0]
    t = -(z[0] + z[1])
    w = z[2]
    return [u, v, t, w]


def from_hex_indices(z):
    v = 2 * z[0] - z[1]
    u = z[1] + z[0]
    w = z[3]
    return [u, v, w]


class DiffractionGenerator(QtWidgets.QWidget):
    def __init__(self, view=None):
        super().__init__()
        self.view = view
        self.setWindowTitle("Diffraction generator")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, on=True)
        # self.setFixedSize(1200, 400)

        self.setStyleSheet(open("expert_pi/gui/style/style.qss", encoding="utf-8").read())
        self.setLayout(QtWidgets.QGridLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.elements = {}

        self.info = QtWidgets.QWidget()
        self.info.setLayout(QtWidgets.QHBoxLayout())
        self.info.layout().setContentsMargins(0, 0, 0, 0)
        self.info.layout().setSpacing(0)

        self.layout().addWidget(self.info, 0, 0, 1, 2)

        self.cif_selector = combo_box.ToolbarComboBox()
        self.cif_path = "data/cifs/"

        self.cifs = []
        self.cif_selector.addItem("")

        self.cif_selector.currentIndexChanged.connect(self.select_cif)
        self.lattice_dimensions = QtWidgets.QLabel("")
        self.zone = QtWidgets.QLineEdit("[1,0,0]")
        self.zone.setFixedWidth(100)
        self.zone.returnPressed.connect(self.goto_zone_clicked)
        self.zone.setToolTip("Type zone in format: [x,y,z] and press enter")
        self.angle = spin_box.SpinBoxWithUnits(
            0, [-180, 180], 10, "deg", decimals=1, send_immediately=True, update_read=True
        )
        self.angle.setToolTip("rotation angle along optical axis")
        self.angle.set_signal.connect(self.gamma_angle_changed)

        self.unit_multiple = combo_box.ToolbarComboBox()
        self.unit_multiple.addItem("unit cell")
        for i in [3, 5]:
            self.unit_multiple.addItem(str(i) + "x")
        self.unit_multiple.setToolTip("multiple unit cell")
        self.unit_multiple.currentIndexChanged.connect(lambda x: self.plot_structure())

        self.info.layout().addWidget(self.cif_selector)
        self.info.layout().addWidget(self.lattice_dimensions)

        self.point_group_button = buttons.ToolbarPushButton("Point group", selectable=True)
        self.point_group_button.setProperty("class", "infoButton")
        self.point_group_button.setToolTip("show visualization of point group")
        self.point_group_button.clicked.connect(self.point_group_button_clicked)

        self.dynamical_button = buttons.ToolbarPushButton("Dynamical")
        self.dynamical_button.setProperty("class", "infoButton")
        self.dynamical_button.setToolTip("generate dynamical structure factors")
        self.dynamical_button.clicked.connect(self.generate_dynamic_factors)

        self.thickness = spin_box.SpinBoxWithUnits(
            0, [0, 10000], 10, "nm", decimals=1, send_immediately=True, update_read=True
        )
        self.thickness.set_signal.connect(self.thickness_changed)
        self.thickness.setEnabled(False)
        self.thickness.setToolTip("thickness of sample, leve zero for kinematic approximation")

        self.advanced = QtWidgets.QWidget()
        self.advanced.setLayout(QtWidgets.QHBoxLayout())
        self.advanced.layout().setContentsMargins(0, 0, 0, 0)
        self.advanced.layout().setSpacing(0)

        self.layout().addWidget(self.advanced, 1, 0, 1, 2)

        self.advanced.layout().addWidget(self.zone)
        self.advanced.layout().addWidget(self.angle)
        self.advanced.layout().addWidget(self.unit_multiple)
        self.advanced.layout().addWidget(self.dynamical_button)
        self.advanced.layout().addWidget(self.thickness)
        self.advanced.layout().addWidget(self.point_group_button)
        self.advanced.layout().addStretch()

        self.orientation_plan_button = buttons.ToolbarPushButton("Orientation plan")
        self.orientation_plan_button.setProperty("class", "infoButton")
        self.orientation_plan_button.setToolTip("generate orientation plan")
        self.orientation_plan_button.clicked.connect(self.generate_orientation_plan)

        self.fit_orientation_button = buttons.ToolbarPushButton("Fit to Image", selectable=True)
        self.fit_orientation_button.setProperty("class", "infoButton")
        self.fit_orientation_button.setToolTip("fit to current diffraction pattern from camera")
        self.fit_orientation_button.setEnabled(False)
        self.fit_orientation_button.clicked.connect(self.fit_orientation)

        self.fit_abs_prominence = spin_box.SpinBoxWithUnits(
            20, [1, 65535], 10, "", decimals=0, send_immediately=True, update_read=True
        )
        self.fit_abs_prominence.setToolTip("Minimal intensity value for detecting spots")
        self.fit_abs_prominence.set_signal.connect(lambda x: self.fit_orientation)
        self.fit_error_radius = spin_box.SpinBoxWithUnits(
            3, [0.01, 150], 1, "mrad", decimals=2, send_immediately=True, update_read=True
        )
        self.fit_error_radius.setToolTip("distance of spots for fitting")
        self.fit_error_radius.set_signal.connect(lambda x: self.fit_orientation)
        self.fit_info = QtWidgets.QLabel("")

        self.fitting_toolbar = QtWidgets.QWidget()
        self.fitting_toolbar.setLayout(QtWidgets.QHBoxLayout())
        self.fitting_toolbar.layout().setContentsMargins(0, 0, 0, 0)
        self.fitting_toolbar.layout().setSpacing(0)

        self.layout().addWidget(self.fitting_toolbar, 2, 0, 1, 2)

        self.fitting_toolbar.layout().addWidget(self.orientation_plan_button)
        self.fitting_toolbar.layout().addWidget(self.fit_orientation_button)
        self.fitting_toolbar.layout().addWidget(self.fit_abs_prominence)
        self.fitting_toolbar.layout().addWidget(self.fit_error_radius)
        self.fitting_toolbar.layout().addWidget(self.fit_info)
        self.fitting_toolbar.layout().addStretch()

        self.real_space = GlViewWidgetOrbit(self.plot_diffraction)
        self.real_space.setCameraParams(distance=1000, fov=1)

        self.legend = GLLegend()
        self.legend.setDepthValue(10000)
        self.real_space.addItem(self.legend)

        self.diffraction_space = pg.PlotWidget()

        self.diffraction_space.setAspectLocked()
        self.diffraction_space.setLabel("left", "mrad")
        self.diffraction_space.setLabel("bottom", "mrad")
        self.diffraction_space.setXRange(-100, 100)
        self.diffraction_space.setYRange(-100, 100)

        self.diffraction_pattern = pg.ScatterPlotItem(
            x=[], y=[], size=[], pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 200)
        )
        self.diffraction_space.addItem(self.diffraction_pattern)

        self.point_group_visualizer = point_group_visualizer.PointGroupVisualizer()
        self.real_space.linked_orbit = self.point_group_visualizer.glview
        self.point_group_visualizer.glview.linked_orbit = self.real_space

        self.layout().setRowStretch(3, 1)
        self.layout().setRowStretch(4, 1)
        self.layout().setColumnStretch(0, 1)
        self.layout().setColumnStretch(1, 2)

        self.layout().addWidget(self.real_space, 3, 0, 1, 1)
        self.layout().addWidget(self.point_group_visualizer, 4, 0, 1, 1)
        self.layout().addWidget(self.diffraction_space, 3, 1, 2, 1)

        # self.energy = self.view.main_window.optics.energy_value if self.view is not None else 100_000
        # TODO update when changes
        self.energy = 100_000
        # self.k_max = 150e-3/(self.view.main_window.optics.get_wavelength()*1e10) if self.view is not None else 6
        self.k_max = 6

        self.progress_threads = {name: None for name in ["dynamical", "orientation_plan"]}

        self.orientation_matrix = np.eye(3)
        self.bragg_peaks = None
        self.cell = None
        self.atoms = []

        self.hide_point_group()

    def reload(self):
        self.cif_selector.blockSignals(True)
        self.cif_selector.clear()

        self.cif_selector.addItem("")
        self.load_cifs()
        for name, file, structure in self.cifs:
            self.cif_selector.addItem(name)

        self.cif_selector.blockSignals(False)

    def load_cifs(self):
        from pymatgen.io.cif import CifParser

        if not os.path.exists(self.cif_path):
            os.makedirs(self.cif_path)

        cifs = [f for f in listdir(self.cif_path) if isfile(join(self.cif_path, f))]
        self.cifs = []
        for file in cifs:
            if file.split(".")[1] != "cif":
                continue
            parser = CifParser(join(self.cif_path, file))
            structure = parser.get_structures()[0]
            self.cifs.append((f"{structure.formula} | {file}", file, structure))

    def select_cif(self, index):
        from pymatgen.symmetry import groups

        self.cif_selector.setStyleSheet("color:red")
        self.repaint()
        if index == 0:
            return
        structure = self.cifs[index - 1][2]

        self.thickness.setEnabled(False)
        self.thickness.setValue(0)

        self.generate_structure()

        self.space_group = structure.get_space_group_info()
        self.point_group = groups.SpaceGroup.from_int_number(self.space_group[1]).point_group

        # fix inconsistencies in space group data:
        translations = {
            "-31m": "-3m",
            "-3m1": "-3m",
            "-4m2": "-42m",
            "-62m": "-6m2",
            "312": "32",
            "31m": "3m",
            "321": "32",
            "3m1": "3m",
        }
        if self.point_group in translations:
            self.point_group = translations[self.point_group]

        if "P6" in self.space_group[0]:
            self.legend.plot_hex = True
        else:
            self.legend.plot_hex = False

        self.lattice_dimensions.setText(
            f"{self.space_group[0]} "
            f" ({self.point_group}) "
            f"<span style='color:red'>{self.structure.cell[0]:5.2f}</span> x "
            f"<span style='color:green'>{self.structure.cell[1]:5.2f}</span> x "
            f"<span style='color:RoyalBlue'>{self.structure.cell[2]:5.2f}</span> Å "
        )

        self.cif_selector.setStyleSheet("")
        self.point_group_visualizer.select_group(self.point_group)

    def point_group_button_clicked(self):
        if self.point_group_button.selected:
            self.show_point_group()
        else:
            self.hide_point_group()

    def hide_point_group(self):
        self.point_group_visualizer.hide()
        self.layout().setRowStretch(4, 0)
        # g = self.geometry()
        # self.setGeometry(g.x(), g.y(), g.width(), int(g.height()/3*2))

    def show_point_group(self):
        self.point_group_visualizer.show()
        self.layout().setRowStretch(4, 1)
        # g = self.geometry()
        # self.setGeometry(g.x(), g.y(), g.width(), int(g.height()/2*3))

    def lattice_to_orientation_matrix(self, zone_axis_lattice, proj_x_lattice=None):
        proj_z = np.array(zone_axis_lattice)
        if proj_z.shape[0] == 4:
            proj_z = self.structure.hexagonal_to_lattice(proj_z)

        proj_z = self.structure.lattice_to_cartesian(proj_z)

        if proj_x_lattice is not None:
            proj_x = np.array(proj_x_lattice)
            if proj_x.shape[0] == 4:
                proj_x = self.structure.hexagonal_to_lattice(proj_x)
            proj_x = self.structure.lattice_to_cartesian(proj_x)
        elif np.abs(proj_z[2]) > 1 - 1e-6:
            proj_x = np.cross(np.array([0, 1, 0]), proj_z)
        else:
            proj_x = np.array([0, 0, -1])

        # Generate orthogonal coordinate system, normalize
        proj_y = np.cross(proj_z, proj_x)
        proj_x = np.cross(proj_y, proj_z)
        proj_x = proj_x / np.linalg.norm(proj_x)
        proj_y = proj_y / np.linalg.norm(proj_y)
        proj_z = proj_z / np.linalg.norm(proj_z)

        return np.vstack((proj_x, proj_y, proj_z)).T

    def rot_matrix_decompose(self, R):
        # Using Euler Angles XYZ convention M=3x3 matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            alpha = np.arctan2(R[2, 1], R[2, 2])
            beta = np.arctan2(-R[2, 0], sy)
            gamma = np.arctan2(R[1, 0], R[0, 0])
        else:  # At Gimble lock
            alpha = np.arctan2(-R[1, 2], R[1, 1])
            beta = np.arctan2(-R[2, 0], sy)
            gamma = 0
        return np.array([alpha, beta, gamma])

    def orientation_matrix_to_lattice(self, orientation_matrix, tol_den=1000):
        proj_x, proj_y, proj_z = orientation_matrix.T
        lattice_vec = self.structure.cartesian_to_lattice(proj_z)
        return self.structure.rational_ind(lattice_vec, tol_den=tol_den)

    def dynamical_progress_updated(self, n, total, start, now):  # mean running from thread
        self.dynamical_button.update_text_signal.emit(f"{n}/{total} {(now - start):4.1f}s ")

    def orientation_plan_progress_updated(self, n, total, start, now):  # mean running from thread
        self.orientation_plan_button.update_text_signal.emit(f"{n}/{total} {(now - start):4.1f}s ")

    def generate_structure(self):
        """max angle mrad"""
        # lazy load:
        import py4DSTEM
        from py4DSTEM.process.diffraction import crystal_bloch

        crystal_bloch._original_tqdm = crystal_bloch.tqdm  # we will patch tqdm to our object after instead

        path = join(self.cif_path, self.cifs[self.cif_selector.currentIndex() - 1][1])
        self.structure = py4DSTEM.process.diffraction.Crystal.from_CIF(path)

        # add matrix orientation to generate_dynamical_diffraction function
        from . import crystal_bloch_patch

        funcType = type(self.structure.generate_dynamical_diffraction_pattern)
        self.structure.generate_dynamical_diffraction_pattern = types.MethodType(
            crystal_bloch_patch.generate_dynamical_diffraction_pattern, self.structure
        )

        self.structure.setup_diffraction(self.energy)

        self.structure.calculate_structure_factors(self.k_max, tol_structure_factor=-1.0)

        # self.structure.match_orientations
        # self.structure.plot_structure
        # self.structure.parse_orientation()
        # self.structure.orientation_plan(zone_axis_range="auto", accel_voltage=energy)
        #
        # d.structure.symmetry_reduce_directions
        # self.structure.plot_orientation_zones()
        # self.structure.orientation_zone_axis_range
        # d.structure.rational_ind(d.structure.cartesian_to_lattice(d.orientation_matrix)[:, 0])

        self.plot_structure()
        M = self.real_space.opts["rotation"].toRotationMatrix()
        self.plot_diffraction(np.array(M.data()).reshape(3, 3))

    def generate_dynamic_factors(self):
        from py4DSTEM.process.diffraction import crystal_bloch

        crystal_bloch.tqdm = tqdm_with_callback(
            self.dynamical_progress_updated
        )  # patch tqdm to output to current widget

        if self.dynamical_button.property("busy"):
            if self.progress_threads["dynamical"] is not None and self.progress_threads["dynamical"].is_alive():

                def on_stop():
                    self.dynamical_button.update_text_signal.emit("Dynamical")
                    self.dynamical_button.setProperty("busy", False)
                    self.dynamical_button.update_style_signal.emit()

                self.progress_threads["dynamical"].try_stop(on_stop)
            return

        self.dynamical_button.setProperty("busy", True)
        self.dynamical_button.update_style_signal.emit()

        def function():
            self.structure.calculate_dynamical_structure_factors(
                self.energy, "WK-CP", k_max=self.k_max, thermal_sigma=0.08, tol_structure_factor=-1.0
            )
            self.dynamical_button.update_text_signal.emit("Dynamical")
            self.dynamical_button.setProperty("busy", False)
            self.thickness.setEnabled(True)
            self.dynamical_button.update_style_signal.emit()

        self.progress_threads["dynamical"] = StoppableThread(target=function)
        self.progress_threads["dynamical"].start()

    def generate_orientation_plan(self):
        from py4DSTEM.utils.tqdmnd import integer, nditer

        def tqdmnd(*args, **kwargs):
            r = [range(i) if isinstance(i, (int, integer)) else i for i in args]
            return tqdm_with_callback(self.orientation_plan_progress_updated)(nditer(*r), **kwargs)

        from py4DSTEM.process.diffraction import crystal_ACOM

        crystal_ACOM.tqdmnd = tqdmnd  # patch tqdmnd to output to current widget

        if self.orientation_plan_button.property("busy"):
            if (
                self.progress_threads["orientation_plan"] is not None
                and self.progress_threads["orientation_plan"].is_alive()
            ):

                def on_stop():
                    self.orientation_plan_button.update_text_signal.emit("Orientation plan")
                    self.orientation_plan_button.setProperty("busy", False)
                    self.orientation_plan_button.update_style_signal.emit()

                self.progress_threads["orientation_plan"].try_stop(on_stop)
            return

        self.orientation_plan_button.setProperty("busy", True)
        self.orientation_plan_button.update_style_signal.emit()

        def function():
            self.structure.orientation_plan(zone_axis_range="auto", accel_voltage=self.energy)
            self.orientation_plan_button.update_text_signal.emit("Orientation plan")
            self.orientation_plan_button.setProperty("busy", False)
            self.fit_orientation_button.setEnabled(True)
            self.orientation_plan_button.update_style_signal.emit()

        self.progress_threads["orientation_plan"] = StoppableThread(target=function)
        self.progress_threads["orientation_plan"].start()

    def fit_orientation(self, relative_prominence=0.01):
        if self.fit_orientation_button.selected:
            N = 512  # assume 512x512 camera image
            image = self.view.raw_image.reshape(N, N)
            angular_fov = self.view.image_item.fov
            minimal_spot_distance = self.fit_error_radius.value() * 2
            absolute_prominence = self.fit_abs_prominence.value()
            orientation, xys, V_max, P = bragg_fitting.fit_orientation(
                image,
                angular_fov,
                self.structure,
                minimal_spot_distance,
                relative_prominence,
                absolute_prominence,
                method="py4DSTEM",
            )

            self.goto_matrix(orientation.matrix[0])

            if self.view.diffraction_pattern.isVisible():
                self.plot_fit(self.bragg_peaks, xys, V_max, minimal_spot_distance / 2, size_factor=0.02)

    def clear(self):
        if self.cell is not None:
            self.real_space.removeItem(self.cell)
        for atom in self.atoms:
            self.real_space.removeItem(atom)
        self.atoms = []

    def generate_units(self, order=1):
        vectors = []
        for i in np.linspace(-order, order, num=2 * order + 1):
            for j in np.linspace(-order, order, num=2 * order + 1):
                for k in np.linspace(-order, order, num=2 * order + 1):
                    vectors.append(np.array([i, j, k]))
        return vectors

    def plot_structure(self, tol_distance=1e-4, size_marker=0.5):
        # lazy load:
        from py4DSTEM.process.diffraction import crystal_viz

        self.clear()

        # unit cell vectors
        u = self.structure.lat_real[0, :]
        v = self.structure.lat_real[1, :]
        w = self.structure.lat_real[2, :]

        # atomic identities
        id_ = self.structure.numbers * 1

        # Fractional atomic coordinates
        pos = self.structure.positions * 1
        if self.unit_multiple.currentText() == "unit cell":
            # x tile
            sub = pos[:, 0] < tol_distance
            pos = np.vstack([pos, pos[sub, :] + np.array([1, 0, 0])])
            id_ = np.hstack([id_, id_[sub]])
            # y tile
            sub = pos[:, 1] < tol_distance
            pos = np.vstack([pos, pos[sub, :] + np.array([0, 1, 0])])
            id_ = np.hstack([id_, id_[sub]])
            # z tile
            sub = pos[:, 2] < tol_distance
            pos = np.vstack([pos, pos[sub, :] + np.array([0, 0, 1])])

            id_ = np.hstack([id_, id_[sub]])
        else:
            if self.unit_multiple.currentText() == "3x":
                vectors = self.generate_units(1)
            elif self.unit_multiple.currentText() == "5x":
                vectors = self.generate_units(2)

            pos = np.vstack([pos + v for v in vectors])
            id_ = np.hstack([id_ for v in vectors])

        # Cartesian atomic positions
        xyz = (pos - np.array([0.5, 0.5, 0.5])) @ self.structure.lat_real

        # 3D plotting

        # unit cell
        p = np.vstack([[0, 0, 0], u, u + v, v, w, u + w, u + v + w, v + w]) - (u + v + w) * 0.5
        p = p[:, [0, 1, 2]]  # Reorder cell boundaries to put normal z on camera view

        f = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        self.p = p
        self.f = f
        colors = np.array([[1.0, 1.0, 1.0]] * 24)
        colors[0, :] = [
            1.0,
            0.0,
            0.0,
        ]  # u
        colors[1, :] = [
            1.0,
            0.0,
            0.0,
        ]  # u
        colors[6, :] = [0.0, 1.0, 0.0]  # v
        colors[7, :] = [0.0, 1.0, 0.0]  # v
        colors[16, :] = [0.0, 0.0, 1.0]  # w
        colors[17, :] = [0.0, 0.0, 1.0]  # w
        self.cell = gl.GLLinePlotItem(pos=p[f].reshape(24, 3), color=colors, width=1, antialias=True, mode="lines")
        self.real_space.addItem(self.cell)

        self.xyz = xyz

        md = gl.MeshData.sphere(rows=32, cols=32)

        for i in range(len(id_)):
            color = crystal_viz.atomic_colors(id_[i])
            rgba = [color[0], color[1], color[2], 1]

            atom = gl.GLMeshItem(meshdata=md, smooth=True, color=rgba, shader="edgeHilight", glOptions="opaque")
            atom.translate(xyz[i][0], xyz[i][1], xyz[i][2])
            atom.scale(size_marker, size_marker, size_marker)

            self.atoms.append(atom)
            self.real_space.addItem(atom)

    def gamma_angle_changed(self, value):
        self.real_space.rotate(-(-self.legend.angles[2] + value))

    def thickness_changed(self):
        M = self.real_space.opts["rotation"].toRotationMatrix()
        self.plot_diffraction(np.array(M.data()).reshape(3, 3))

    def goto_zone_clicked(self):
        text = self.zone.text()
        try:
            numbers = text.split("[")[1].split("]")[0].split(",")
            self.zone.setStyleSheet("")
            zone = [int(n) for n in numbers[:3]]
        except:
            self.zone.setStyleSheet("color:red")
            return None
        self.set_zone_axis(zone, angle=self.angle.value())

    def show(self):
        super().show()
        if self.view is not None:
            self.view.diffraction_pattern.show()

    def closeEvent(self, e):
        if self.view is not None:
            self.view.diffraction_pattern.hide()

    def set_zone_axis(self, zone_axis_lattice, angle=0.0):
        M = self.lattice_to_orientation_matrix(zone_axis_lattice)
        self.goto_matrix(M, angle)

    def goto_matrix(self, M, angle=0.0):
        q = QtGui.QQuaternion.fromRotationMatrix(QtGui.QMatrix3x3(M.T.flatten()))
        self.real_space.set_quaterion(q)  # trigger plotting

        if angle != 0:
            self.real_space.rotate(angle)

        q = self.real_space.opts["rotation"]
        angles = list(q.toEulerAngles().toTuple())
        self.angle.set_read_sent_value(angles[2])

    def plot_diffraction(self, M, disk_radius=1, size_factor=0.2):
        """
        :param M: 3x3 rotation amatrix
        :param disk_radius: mrad
        """
        zone_axis_lattice = self.orientation_matrix_to_lattice(M, tol_den=5)
        q = QtGui.QQuaternion.fromRotationMatrix(QtGui.QMatrix3x3(M.flatten()))
        angles = list(q.toEulerAngles().toTuple())

        self.legend.update_info(zone_axis_lattice, angles)
        self.orientation_matrix = M
        self.bragg_peaks_kinematic = self.structure.generate_diffraction_pattern(
            orientation_matrix=M, sigma_excitation_error=0.02, tol_intensity=0
        )

        if self.thickness.value() > 0:
            self.bragg_peaks = self.structure.generate_dynamical_diffraction_pattern(
                beams=self.bragg_peaks_kinematic,
                thickness=self.thickness.value() * 10,
                orientation_matrix=M,  # nm to A
            )
        else:
            self.bragg_peaks = self.bragg_peaks_kinematic
        A_to_mrad = 1000 * self.structure.wavelength
        self.A_to_mrad = A_to_mrad
        x = [a[0] * A_to_mrad for a in self.bragg_peaks.data]
        y = [a[1] * A_to_mrad for a in self.bragg_peaks.data]
        zone = np.array([[a[3], a[4], a[5]] for a in self.bragg_peaks.data])

        zero_index = np.argwhere(np.all(zone == np.array([0, 0, 0]), axis=1))[0][0]

        i = np.array([a[2] for a in self.bragg_peaks.data])
        i[zero_index] = 0
        i **= 0.2
        i /= np.max(i)
        brushes = [pg.mkBrush(255, 255, 255, 255 * i) for i in i]
        brushes[zero_index] = pg.mkBrush(255, 100, 0, 255)

        s = np.array([np.array(a[2]) ** size_factor for a in self.bragg_peaks.data])

        size = 20
        min_size = 0
        s2 = s / s[zero_index] * size
        s2 = np.maximum(min_size, s2)

        self.diffraction_pattern.setData(x=x, y=y, size=disk_radius * 2, brush=brushes, pxMode=False)

        if self.view is not None and self.view.diffraction_pattern.isVisible():
            self.view.diffraction_pattern.generate_model(np.array([x, y, s2]).T)

            if self.fit_orientation_button.selected:
                n = 512  # assume 512x512 camera image
                image = self.view.raw_image.reshape(n, n)
                angular_fov = self.view.image_item.fov
                minimal_spot_distance = 2 * self.fit_error_radius.value()
                relative_prominence = 0.01
                absolute_prominence = self.fit_abs_prominence.value()
                xys, v_max, _ = bragg_fitting.fit_diffraction_patterns(
                    np.array([image]), angular_fov, minimal_spot_distance, relative_prominence, absolute_prominence
                )[0]

                r2 = xys[:, 0] ** 2 + xys[:, 1] ** 2
                self.plot_fit(self.bragg_peaks, xys, v_max * r2, minimal_spot_distance / 2, size_factor=size_factor)

    def plot_fit(self, bragg_peaks, xys, i_s, angular_error, size_factor=0.2):
        qxy = xys / 1000 / self.structure.wavelength  # to reciprocal A

        qx = bragg_peaks.data["qx"]
        qy = bragg_peaks.data["qy"]
        q_i = bragg_peaks.data["intensity"]

        k_error = angular_error / self.A_to_mrad

        _q_i2, p_i2 = bragg_fitting.highlight_diffraction_by_fitting(qx, qy, q_i, qxy[:, 0], qxy[:, 1], i_s, k_error)

        self.fit_info.setText(f"{np.sum(p_i2):10.3f}")
        size = p_i2**size_factor
        size = size / np.max(size) * angular_error

        if self.view is not None and self.view.diffraction_pattern.isVisible():
            self.view.diffraction_pattern.generate_fit(
                qxy[:, 0] * self.A_to_mrad, qxy[:, 1] * self.A_to_mrad, size, angular_error
            )
