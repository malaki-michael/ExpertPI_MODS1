import numpy as np
import pymatgen.core as _pmg  # noqa: F401, TODO: circular import bug in pymatgen
from pymatgen.symmetry import groups
from pyqtgraph import opengl as gl
from PySide6 import QtWidgets

from expert_pi.gui.tools.gl_view import GlViewWidgetOrbit
from expert_pi.measurements.orientation import point_group_mesh_generator as mesher


def generate_arc(v0, v1, angle_step=0.02):
    vx = v0 / np.sqrt(np.sum(v0**2))
    vy = v1 / np.sqrt(np.sum(v1**2))
    angle = np.arccos(np.inner(vx, vy))

    angles = np.linspace(0, angle, num=int(angle / angle_step))
    points = np.array([np.cos(angles), np.sin(angles), 0 * angles])

    vy = vy - np.inner(vx, vy) * vx
    vy = vy / np.sqrt(np.sum(vy**2))

    vz = np.cross(vx, vy)
    vz = vz / np.sqrt(np.sum(vz**2))

    R = np.vstack([vx, vy, vz]).T

    return R @ points


hexagonal_point_groups = ["-3", "-3m", "-6", "-6m2", "3", "32", "3m", "6", "6/m", "6/mmm", "622", "6mm"]

hex_to_cartesian = np.array([[1, -1 / 2, 0], [0, np.sqrt(3) / 2, 0], [0, 0, 1]])

cartesian_to_hex = np.linalg.inv(hex_to_cartesian)


def reduce_points_to_group(points, point_group, tolerance_decimals=5):
    points_orbits = get_orbits(points.T, point_group, tolerance_decimals=tolerance_decimals, filter=False)
    points_orbits = np.array(points_orbits)
    reduced_points = [np.unique(points_orbits[:, :, i], axis=0)[-1] for i in range(points_orbits.shape[2])]
    return np.array(reduced_points)


def get_orbits(cartesian_vector, point_group, tolerance_decimals=5, filter=True):
    if point_group.symbol in hexagonal_point_groups:
        # we need to compensate for the definition of hexagonal non-carthesian system:
        vector = cartesian_to_hex @ cartesian_vector
        hexagonal = True
    else:
        vector = cartesian_vector
        hexagonal = False
    orbits = []
    for o in point_group.symmetry_ops:
        orbit = o.rotation_matrix @ vector
        if hexagonal:
            orbit = hex_to_cartesian @ orbit
        orbits.append(orbit)
    if filter:
        orbits = np.unique(np.array(orbits).round(decimals=tolerance_decimals), axis=0)
    return orbits


class PointGroupVisualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setLayout(QtWidgets.QGridLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.glview = GlViewWidgetOrbit(self.set_rotation_matrix)
        self.glview.setCameraParams(distance=200, fov=1)

        # self.group_selector = QtWidgets.QComboBox()
        # for name in mesher.orientation_ranges.keys():
        #     self.group_selector.addItem(name)
        #
        # self.group_selector.currentTextChanged.connect(self.select_group)
        #
        # self.widget.layout().addWidget(self.group_selector, 0, 0, 1, 1)

        self.layout().addWidget(self.glview, 1, 0, 1, 1)

        self.items = []
        self.color_mesh = []

        self.point_group = groups.PointGroup("1")
        self.plot_sphere()

    def select_group(self, name):
        self.point_group = groups.PointGroup(name)
        self.clear()
        self.plot_sphere()

    def clear(self):
        for item in self.items:
            self.glview.removeItem(item)

        for mesh in self.color_mesh:
            self.glview.removeItem(mesh)

        for point in self.points:
            self.glview.removeItem(point)

        self.color_mesh = []
        self.items = []
        self.points = []

    def get_color(self, M):
        proj_z = R[:, 2]
        orbits = get_orbits(proj_z, self.point_group, tolerance_decimals=5)

    def plot_sphere(self):
        vectors = mesher.orientation_ranges[self.point_group.symbol]
        F = 1.01
        N = 30

        for i in range(len(vectors)):
            i2 = i + 1
            if i2 == len(vectors):
                i2 = 0

            points = generate_arc(np.array(vectors[i]), np.array(vectors[i2]))

            circles = get_orbits(points, self.point_group, filter=False)

            for circle in circles:
                self.items.append(
                    gl.GLLinePlotItem(
                        pos=F * circle.T,
                        color=(0, 0, 0, 1),
                        width=1,
                        antialias=True,
                        mode="line_strip",
                        glOptions="opaque",
                    )
                )
            # vector_norm = np.array(vectors[i])/np.sqrt(np.sum(np.array(vectors[i])**2))
            # text = gl.GLTextItem(pos=vector_norm*1.1, text=str(vectors[i]), glOptions="opaque")
            # text.setDepthValue(100)
            # self.items.append(text)

        # ------------------------------
        points, edges, triangles, colors = mesher.get_mesh(
            [mesher.normalize(v) for v in vectors], angular_step=10 / 180 * np.pi
        )
        self.plot_color_mesh(points, edges, triangles, colors)

        # ------------------------------
        # md = gl.MeshData.sphere(rows=N, cols=N)
        #
        # colors = []
        # print(md._vertexes.shape[0])
        # for i in range(md._vertexes.shape[0]):
        #     reduced = get_orbits(md._vertexes[i], self.point_group, tolerance_decimals=5)[-1]
        #     reduced /= np.sqrt(np.sum(reduced**2))
        #     if np.sum(np.abs(reduced - md._vertexes[i])) < 1e-3:
        #         colors.append([0, 0.5, 0, 0.2])
        #     else:
        #         colors.append([0.1, 0.1, 0.1, 0.2])
        #
        # md.setVertexColors(colors)
        #
        # self.items.append(gl.GLMeshItem(meshdata=md, smooth=True, color=(0., 0, 0, 0), computeNormals=False, glOptions="opaque"))
        # self.items[-1].scale(0.95, 0.95, 0.95)
        # # self.items.append(gl.GLScatterPlotItem(pos=[[1, 0, 0]], color=(1, 0, 0, 1), size=10, shader="distance", glOptions="opaque"))
        # self.actual = gl.GLScatterPlotItem(pos=[[1, 0, 0]], color=(1, 1, 1, 1), size=10, glOptions="opaque")
        # self.items.append(self.actual)

        self.points = []

        # axes labels
        for v in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            text = gl.GLTextItem(pos=np.array(v) * 1.1, text=str(v), glOptions="opaque")
            text.setDepthValue(100)
            self.items.append(text)

        for item in self.items:
            self.glview.addItem(item)

    def plot_color_mesh(self, points, edges, triangles, colors):
        for mesh in self.color_mesh:
            self.glview.removeItem(mesh)
        self.color_mesh = []

        faces = np.sort(edges[triangles].reshape(-1, 6), axis=1)[:, ::2]
        rgba = np.hstack([colors, 255 * np.ones((colors.shape[0], 1))])

        circles = get_orbits(points.T, self.point_group, filter=False)

        for circle in circles:
            md = gl.MeshData(vertexes=circle.T, faces=faces, edges=None, vertexColors=rgba / 255, faceColors=None)
            self.color_mesh.append(
                gl.GLMeshItem(meshdata=md, smooth=True, color=(1, 0, 0, 1), computeNormals=False, glOptions="opaque")
            )
            self.glview.addItem(self.color_mesh[-1])

    def set_rotation_matrix(self, R):
        for p in self.points:
            self.glview.removeItem(p)

        proj_z = R[:, 2]
        orbits = get_orbits(proj_z, self.point_group, tolerance_decimals=5)

        N = 32

        md = gl.MeshData.sphere(rows=N, cols=N)
        md._vertexes *= 0.03
        self.points.append(
            gl.GLMeshItem(meshdata=md, smooth=True, color=[1, 0, 0, 1], computeNormals=False, glOptions="translucent")
        )
        self.points[-1].translate(1, 0, 0)
        self.glview.addItem(self.points[-1])
        self.points.append(
            gl.GLMeshItem(meshdata=md, smooth=True, color=[0, 1, 0, 1], computeNormals=False, glOptions="translucent")
        )
        self.points[-1].translate(0, 1, 0)
        self.glview.addItem(self.points[-1])
        self.points.append(
            gl.GLMeshItem(meshdata=md, smooth=True, color=[0, 0, 1, 1], computeNormals=False, glOptions="translucent")
        )
        self.points[-1].translate(0, 0, 1)
        self.glview.addItem(self.points[-1])

        self.points = []
        for i in range(len(orbits)):
            md = gl.MeshData.sphere(rows=N, cols=N)
            md._vertexes *= 0.03
            md._vertexes += orbits[i]

            self.points.append(
                gl.GLMeshItem(
                    meshdata=md, smooth=True, color=[0, 0, 0, 1], computeNormals=False, glOptions="translucent"
                )
            )
            self.glview.addItem(self.points[-1])


#
# widget = DiffractionGenerator()
# widget.show()
