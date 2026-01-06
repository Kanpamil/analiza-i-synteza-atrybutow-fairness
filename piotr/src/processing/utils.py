import numpy as np
import scipy as sp
import pandas as pd
from processing.constants import ROUND_DECIMALS, FORMAT_DECIMALS

def get_hovers(sim_data):
    assert sim_data.weights.shape[0] == sim_data.colors.shape[0]
    assert sim_data.weights.shape[1] == len(sim_data.labels)

    template = '<br>'.join(f'{l}: {{:.{FORMAT_DECIMALS}f}}' for l in sim_data.labels) \
               + f'<br>Value: {{:.{FORMAT_DECIMALS}f}}'

    return [template.format(*w, c) for w, c in zip(sim_data.weights, sim_data.colors)]

def get_aggregated_verts_labels(sim_data):
    verts_cols = [f'v{i}' for i in range(sim_data.verts.shape[1])]

    df = pd.DataFrame(sim_data.verts, columns=verts_cols)
    df['labels'] = sim_data.labels

    df = df.round({col: ROUND_DECIMALS for col in verts_cols})

    df = df.groupby(verts_cols, as_index=False, sort=False).agg({'labels': ', '.join})

    return df[verts_cols].to_numpy(), df['labels'].to_list()

def get_shifted_verts(sim_data, factor=0.1):
    centroid = sim_data.verts.mean(axis=0)
    directions = sim_data.verts - centroid

    return sim_data.verts + (directions * factor)

def get_edges(sim_data):
    hull = sp.spatial.ConvexHull(sim_data.verts)
    n_dim = sim_data.verts.shape[1]
    nan_spacer = [np.nan] * n_dim
    edges = []

    if n_dim == 2:
        for (v1, v2) in hull.simplices:
            edges.append(sim_data.verts[v1])
            edges.append(sim_data.verts[v2])
            edges.append(nan_spacer)
    elif n_dim == 3:
        for i in range(hull.neighbors.shape[0]):
            for j in range(hull.neighbors.shape[1]):
                k = hull.neighbors[i, j]

                if k > i:
                    normal_i = hull.equations[i, :-1]
                    normal_k = hull.equations[k, :-1]

                    if not np.isclose(np.dot(normal_i, normal_k), 1.0):
                        v1 = hull.simplices[i, (j + 1) % n_dim]
                        v2 = hull.simplices[i, (j + 2) % n_dim]

                        edges.append(sim_data.verts[v1])
                        edges.append(sim_data.verts[v2])
                        edges.append(nan_spacer)
    else:
        raise NotImplementedError('Only supported for 2D and 3D data')

    return np.array(edges)
