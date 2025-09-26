import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

cmap = np.load(os.path.dirname(__file__) + "/ziegler.npy")


# cmap = np.load(os.path.dirname(__file__) + "/explore_colormap.npy")


@njit
def project_to_3d_triangle(points, v0, v1, v2):
    data = []
    for r in points:
        M = np.array([
            [v1[0] - v0[0], v2[0] - v0[0], -r[0]],
            [v1[1] - v0[1], v2[1] - v0[1], -r[1]],
            [v1[2] - v0[2], v2[2] - v0[2], -r[2]],
        ])
        c = np.linalg.solve(M, -v0)
        data.append(c)
    return data


def to_square_colors(projections, basis=np.array([[0, 0], [1, 1], [0, 1]])):
    xy = basis[0] + projections[:, 0].reshape(-1, 1) * basis[1] + projections[:, 1].reshape(-1, 1) * basis[2]

    ij = np.clip(np.round(xy * (cmap.shape[0] - 1)).astype("int"), 0, cmap.shape[0] - 1)  # assuming square of cmap

    return cmap[ij[:, 0], ij[:, 1]]


def get_colors(orientations, v0, v1, v2, flip=False):
    projections = project_to_3d_triangle(orientations, v0, v1, v2)

    basis = np.array([[0, 0], [1, 1], [0, 1]])
    if flip:
        basis = np.array([[0, 0], [1, 1], [1, 0]])

    return to_square_colors(np.array(projections), basis)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_colors_half(orientations, v0, v1):
    u = np.sum(orientations * v0, axis=1)
    v = np.sum(orientations * v1, axis=1)

    x, y = disc_to_square(u, v)

    xi = np.clip(np.round((x / 2 + 0.5) * (cmap.shape[0] - 1)).astype("int"), 0, cmap.shape[1] - 1)
    yi = np.clip(np.round((y / 2 + 0.5) * (cmap.shape[1] - 1)).astype("int"), 0, cmap.shape[1] - 1)

    return cmap[xi, yi]


def get_colors_full(orientations):
    z = np.sign(orientations[:, 2]) * orientations[:, 2] ** 2
    a = np.arctan2(orientations[:, 0], orientations[:, 1]) / np.pi
    xy = (
        np.array([0.5, 0.5])
        + z.reshape(-1, 1) * np.array([-0.5, 0.5])
        + ((1 - np.abs(z)) * a).reshape(-1, 1) * np.array([0.5, 0.5])
    )
    xi = np.clip(np.round((xy[:, 0]) * (cmap.shape[0] - 1)).astype("int"), 0, cmap.shape[1] - 1)
    yi = np.clip(np.round((xy[:, 1]) * (cmap.shape[1] - 1)).astype("int"), 0, cmap.shape[1] - 1)
    return cmap[xi, yi]


def subdivide(points, edges, triangles):
    pt = points[edges]
    lengths = np.arccos(np.sum(pt[:, 0, :] * pt[:, 1, :], axis=1))
    i_max = np.argmax(lengths)

    # print("cutting",i_max)

    i_triangles = np.argwhere(triangles == i_max)
    to_delete_triangles_indices = i_triangles[:, 0]
    to_delete_triangles = triangles[to_delete_triangles_indices]

    # add point in the middle:
    new_point = np.mean(points[edges[i_max]], axis=0)
    new_point = new_point / np.sqrt(np.sum(new_point**2))
    points = np.vstack([points, [new_point]])
    i_point = len(points) - 1

    mask = to_delete_triangles != i_max
    remaining_edges = to_delete_triangles[mask].reshape(-1, 2)

    mask_remaining_point = (edges[to_delete_triangles] != edges[i_max][0]) & (
        edges[to_delete_triangles] != edges[i_max][1]
    )
    remaining_points_indices = edges[to_delete_triangles][mask_remaining_point][::2]

    rp = remaining_points_indices[0]
    re = remaining_edges[0]
    N = len(edges)

    if edges[re[0]][0] == edges[re[1]][0]:
        pass
    elif edges[re[0]][0] == edges[re[1]][1]:
        edges[re[1]] = edges[re[1]][::-1]
    elif edges[re[0]][1] == edges[re[1]][0]:
        edges[re[0]] = edges[re[0]][::-1]
    else:
        edges[re[0]] = edges[re[0]][::-1]
        edges[re[1]] = edges[re[1]][::-1]
    edges = np.vstack([
        edges,
        [
            [edges[re[0]][0], i_point],  # middle cut
            [edges[re[0]][1], i_point],
            [edges[re[1]][1], i_point],
        ],
    ])
    triangles = np.vstack([triangles, [[re[0], N, N + 1], [re[1], N, N + 2]]])
    if len(remaining_points_indices) == 2:
        rp = remaining_points_indices[1]
        re = remaining_edges[1]

        if edges[re[0]][0] == edges[re[1]][0]:
            pass
        elif edges[re[0]][0] == edges[re[1]][1]:
            edges[re[1]] = edges[re[1]][::-1]
        elif edges[re[0]][1] == edges[re[1]][0]:
            edges[re[0]] = edges[re[0]][::-1]
        else:
            edges[re[0]] = edges[re[0]][::-1]
            edges[re[1]] = edges[re[1]][::-1]

        edges = np.vstack([edges, [[edges[re[0]][0], i_point]]])

        if edges[-2][0] == edges[re[0]][1]:
            triangles = np.vstack([triangles, [[re[0], N + 3, N + 2], [re[1], N + 3, N + 1]]])
        else:
            triangles = np.vstack([triangles, [[re[0], N + 3, N + 1], [re[1], N + 3, N + 2]]])

    triangles[triangles > i_max] -= 1
    edges = np.delete(edges, i_max, axis=0)

    for i in np.sort(to_delete_triangles_indices)[::-1]:
        triangles = np.delete(triangles, i, axis=0)

    N = len(remaining_points_indices)

    # plot_mesh(points,edges,triangles)
    return points, edges, triangles


def disc_to_square(u, v):
    x = 0.5 * np.sqrt(2 + u**2 - v**2 + 2 * np.sqrt(2) * u) - 0.5 * np.sqrt(2 + u**2 - v**2 - 2 * np.sqrt(2) * u)
    y = 0.5 * np.sqrt(2 - u**2 + v**2 + 2 * np.sqrt(2) * v) - 0.5 * np.sqrt(2 - u**2 + v**2 - 2 * np.sqrt(2) * v)
    return x, y


def subdivide_limit_step(points, edges, triangles, max_angular_length, max_iterations=100_000):
    for i in range(max_iterations):
        pt = points[edges]
        lengths = np.arccos(np.sum(pt[:, 0, :] * pt[:, 1, :], axis=1))
        angular_length = np.max(lengths)
        if np.isnan(angular_length):
            break
        if angular_length < max_angular_length:
            break
        points, edges, triangles = subdivide(points, edges, triangles)
    return points, edges, triangles


def get_normals_to_plane(points, a, b):
    return points - np.sum(points * a, axis=1).reshape(-1, 1) * a - np.sum(points * b, axis=1).reshape(-1, 1) * b


def map_points_to_colors(points, vectors):
    """vectors needs to be normalized points within vector range"""
    vectors = np.array(vectors).astype("float")
    if len(vectors) == 0:
        colors = get_colors_full(points)
    elif len(vectors) == 3:
        colors = get_colors(points, vectors[1], vectors[2], vectors[0])
    elif len(vectors) == 4:
        if np.sum(np.abs(vectors[1] + vectors[3])) == 0:
            colors = get_colors_half(points, vectors[0], vectors[1])
        else:
            # assuming triangles 0,1,3 and 1,3,2
            normals = get_normals_to_plane(points, vectors[1], vectors[3])
            v0_normal = get_normals_to_plane(np.array([vectors[0]]), vectors[1], vectors[3])[0]
            mask_flip = np.sum(normals * v0_normal, axis=1) < 0

            colors = get_colors(points, vectors[1], vectors[3], vectors[0])
            if np.sum(mask_flip) > 0:
                colors[mask_flip, :] = get_colors(points[mask_flip, :], vectors[1], vectors[3], vectors[2], flip=True)
    return colors


def get_mesh(vectors, angular_step=5 / 180 * np.pi, max_iterations=100_000):
    """vectors needs to be normalized"""
    vectors = np.array(vectors).astype("float")
    if len(vectors) == 0:
        # no symmetry
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        edges = np.array([
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 5],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [1, 2],
            [2, 4],
            [4, 5],
            [5, 1],
        ])
        triangles = np.array([
            [0, 1, 8],
            [1, 2, 9],
            [2, 3, 10],
            [3, 0, 11],
            [4, 5, 8],
            [5, 6, 9],
            [6, 7, 10],
            [7, 4, 11],
        ])

        points, edges, triangles = subdivide_limit_step(points, edges, triangles, angular_step, max_iterations)
        colors = get_colors_full(points)
        return points, edges, triangles, colors

    if len(vectors) == 3:
        points0 = np.array([vectors[0], vectors[1], vectors[2]])
        edges0 = np.array([[0, 1], [1, 2], [2, 0]])
        triangles0 = np.array([[0, 1, 2]])

        points, edges, triangles = subdivide_limit_step(points0, edges0, triangles0, angular_step, max_iterations)
        colors = get_colors(points, vectors[1], vectors[2], vectors[0])
        return points, edges, triangles, colors

    elif len(vectors) == 4:
        points0 = np.array([vectors[0], vectors[1], vectors[3]])
        edges0 = np.array([[0, 1], [1, 2], [2, 0]])
        triangles0 = np.array([[0, 1, 2]])

        half = False
        if np.sum(np.abs(vectors[1] + vectors[3])) == 0:
            print("half detected")
            half = True

        if half:
            points0, edges0, triangles0 = modify_half_sphere(points0, vectors[0], vectors[1])

        points, edges, triangles = subdivide_limit_step(points0, edges0, triangles0, angular_step, max_iterations)
        if half:
            colors = get_colors_half(points, points0[0], points0[1])
        else:
            colors = get_colors(points, vectors[1], vectors[3], vectors[0])

        # second triangle:
        points0 = np.array([vectors[2], vectors[3], vectors[1]])
        edges0 = np.array([[0, 1], [1, 2], [2, 0]])
        triangles0 = np.array([[0, 1, 2]])

        if half:
            points0, edges0, triangles0 = modify_half_sphere(points0, vectors[0], vectors[1])

        points2, edges2, triangles2 = subdivide_limit_step(points0, edges0, triangles0, angular_step)
        if half:
            colors2 = get_colors_half(points2, vectors[0], vectors[1])
        else:
            colors2 = get_colors(points2, vectors[1], vectors[3], vectors[2], flip=True)

        # return points2, edges2, triangles2,colors2

        return (
            np.vstack([points, points2]),
            np.vstack([edges, edges2 + len(points)]),
            np.vstack([triangles, triangles2 + len(edges)]),
            np.vstack([colors, colors2]),
        )


def modify_half_sphere(vectors, v0, v1):
    points_m = np.array([vectors[0], vectors[1], np.cross(vectors[0], vectors[1]), vectors[2]])
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]])
    triangles = np.array([[0, 1, 2], [2, 3, 4]])
    return points_m, edges, triangles


orientation_ranges = {
    "1": [],
    "-1": [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]],
    "2": [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]],
    "m": [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]],
    "2/m": [[0, 0, 1], [0, 1, 0], [0, 0, -1], [1, 0, 0]],
    "222": [[0, 0, 1], [0, 1, 0], [0, 0, -1], [1, 0, 0]],
    "mm2": [[0, 0, 1], [0, 1, 0], [0, 0, -1], [1, 0, 0]],
    "mmm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "4": [[0, 0, 1], [1, 1, 0], [0, 0, -1], [1, -1, 0]],
    "-4": [[0, 0, 1], [1, 1, 0], [0, 0, -1], [1, -1, 0]],
    "4/m": [[0, 0, 1], [1, 1, 0], [1, -1, 0]],
    "422": [[0, 0, 1], [1, 1, 0], [0, 0, -1], [1, 0, 0]],
    "4mm": [[0, 0, 1], [1, 1, 0], [0, 0, -1], [1, 0, 0]],
    "-42m": [[0, 0, 1], [1, 1, 0], [0, 0, -1], [1, 0, 0]],
    "4/mmm": [[1, 0, 0], [0, 0, 1], [1, 1, 0]],
    "3": [[0, 0, 1], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, -1], [0.5, -0.5 * np.sqrt(3), 0]],
    "-3": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [0.5 * np.sqrt(3), -0.5, 0]],
    "32": [[0, 0, -1], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, 1], [1, 0, 0]],
    "3m": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [0.5 * np.sqrt(3), -0.5, 0]],
    "-3m": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [1, 0, 0]],
    "6": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [0.5 * np.sqrt(3), -0.5, 0]],
    "-6": [[0, 0, 1], [0.5, 0.5 * np.sqrt(3), 0], [0.5, -0.5 * np.sqrt(3), 0]],
    "6/m": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0.5 * np.sqrt(3), -0.5, 0]],
    "622": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [1, 0, 0]],
    "6mm": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, -1], [1, 0, 0]],
    "-6m2": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [0.5 * np.sqrt(3), -0.5, 0]],
    "6/mmm": [[0, 0, 1], [0.5 * np.sqrt(3), 0.5, 0], [1, 0, 0]],
    "23": [[1, 0, 1], [1, 1, 1], [1, 1, -1], [1, 0, -1]],
    "m-3": [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
    "432": [[1, 0, 0], [1, 1, 1], [1, 1, -1]],
    "-43m": [[1, 0, 0], [1, 1, 1], [1, 1, -1]],
    "m-3m": [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
}

if __name__ == "__main__":
    i = 0
    print(list(orientation_ranges.keys())[i])
    vectors = orientation_ranges[list(orientation_ranges.keys())[i]]
    vectors = [normalize(v) for v in vectors]

    points, edges, triangles, colors = get_mesh(vectors, angular_step=45 / 180 * np.pi, max_iterations=10)

    print(len(points))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], facecolor=colors / 255)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    for edge in edges:
        ax.plot(points[edge, 0], points[edge, 1], points[edge, 2], color="gray")
    plt.show()
