import numpy as np
import scipy as sp
import itertools

# https://mathoverflow.net/a/38736
def get_simplex_vertices(n):
    H = sp.linalg.hadamard(n)
    return H[:, 1:]

# https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Edge_approach
def to_cartesian(weights, vertices):
    return np.matmul(weights, vertices)

# https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Vertex_approach
def to_barycentric(coords, vertices):
    n = vertices.shape[0]
    R = np.vstack([np.ones((1, n)), vertices.T])
    r = np.vstack([np.ones((1, coords.shape[0])), coords.T])

    return np.linalg.solve(R, r).T

# https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)#Example_2
def generate_simplex_grid(n, res):
    assert n >= 1
    assert res >= 1

    weights = []

    for bar_placements in itertools.combinations_with_replacement(range(res + 1), n - 1):
        padded = (0,) + bar_placements + (res,)
        distances = np.diff(padded)
        weights.append(distances)

    weights = np.array(weights) / res
    verts = get_simplex_vertices(n)
    coords = to_cartesian(weights, verts)

    return verts, coords, weights
