import numpy as np
import scipy as sp
import itertools

# https://mathoverflow.net/a/38736
def generate_vertices_cartesian(n_vertices):
    H = sp.linalg.hadamard(n_vertices)
    return H[:, 1:].astype(np.float64)

# https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Edge_approach
def to_cartesian(points_barycentric, vertices_cartesian):
    return np.matmul(points_barycentric, vertices_cartesian, dtype=np.float64)

# https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Vertex_approach
def to_barycentric(points_cartesian, vertices_cartesian):
    n_vertices = vertices_cartesian.shape[0]
    n_points = points_cartesian.shape[0]

    R = np.vstack([np.ones((1, n_vertices)), vertices_cartesian.T], dtype=np.float64)
    r = np.vstack([np.ones((1, n_points)), points_cartesian.T], dtype=np.float64)

    return np.linalg.solve(R, r).T

# https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)#Example_2
def generate_samples(n_vertices, resolution):
    points_barycentric = []

    for placements in itertools.combinations_with_replacement(range(resolution + 1), n_vertices - 1):
        padded = (0,) + placements + (resolution,)
        distances = np.diff(padded)
        points_barycentric.append(distances)

    points_barycentric = np.array(points_barycentric, dtype=np.float64) / resolution

    vertices_cartesian = generate_vertices_cartesian(n_vertices)
    points_cartesian = to_cartesian(points_barycentric, vertices_cartesian)

    return vertices_cartesian, points_cartesian, points_barycentric
