import numpy as np
import scipy as sp
from processing.constants import FORMAT_DECIMALS

def create_hovers(sim_data):
    vertex_labels = sim_data.vertex_labels
    component_names = sim_data.metrics_class.component_names()

    if np.isin(vertex_labels, component_names).all():
        labels = vertex_labels
    else:
        labels = component_names

    template = '<br>'.join(f'{l}: {{:.{FORMAT_DECIMALS}f}}' for l in labels) + f'<br>Value: {{:.{FORMAT_DECIMALS}f}}'

    return np.array([template.format(*b, v) for b, v in zip(sim_data.points_barycentric, sim_data.point_values)])

def create_offset_vertices(sim_data, factor=0.1):
    centroid = sim_data.vertices_cartesian.mean(axis=0)
    directions = sim_data.vertices_cartesian - centroid

    return sim_data.vertices_cartesian + (directions * factor)

def create_wireframe_2d(sim_data):
    hull = sp.spatial.ConvexHull(sim_data.vertices_cartesian)
    nan_spacer = [np.nan, np.nan]
    edges = []

    for (v1, v2) in hull.simplices:
        edges.append(sim_data.vertices_cartesian[v1])
        edges.append(sim_data.vertices_cartesian[v2])
        edges.append(nan_spacer)

    return np.array(edges, dtype=np.float64)

def create_wireframe_3d(sim_data):
    hull = sp.spatial.ConvexHull(sim_data.vertices_cartesian)
    nan_spacer = [np.nan, np.nan, np.nan]
    edges = []

    for i in range(hull.neighbors.shape[0]):
        for j in range(hull.neighbors.shape[1]):
            k = hull.neighbors[i, j]

            if k > i:
                normal_i = hull.equations[i, :-1]
                normal_k = hull.equations[k, :-1]

                if not np.isclose(np.dot(normal_i, normal_k), 1.0):
                    v1 = hull.simplices[i, (j + 1) % 3]
                    v2 = hull.simplices[i, (j + 2) % 3]

                    edges.append(sim_data.vertices_cartesian[v1])
                    edges.append(sim_data.vertices_cartesian[v2])
                    edges.append(nan_spacer)

    return np.array(edges, dtype=np.float64)
