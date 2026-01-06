import copy
import numpy as np
import pandas as pd
from core.simplex import get_simplex_vertices, to_cartesian
from processing.constants import ROUND_DECIMALS

def slice_dimensions(sim_data, constraint_to_value):
    assert isinstance(constraint_to_value, dict)

    sim_data_dcopy = copy.deepcopy(sim_data)

    for constraint, value in constraint_to_value.items():
        index = sim_data_dcopy.labels.index(constraint)
        mask = sim_data_dcopy.weights[:, index] == value

        sim_data_dcopy.weights = sim_data_dcopy.weights[mask]
        sim_data_dcopy.colors = sim_data_dcopy.colors[mask]

    indices_to_keep = [i for i in range(len(sim_data_dcopy.labels))
                       if sim_data_dcopy.labels[i] not in constraint_to_value.keys()]

    sim_data_dcopy.labels = [sim_data_dcopy.labels[i] for i in indices_to_keep]
    sim_data_dcopy.verts = get_simplex_vertices(len(sim_data_dcopy.labels))

    sim_data_dcopy.weights = sim_data_dcopy.weights[:, indices_to_keep]
    sim_data_dcopy.coords = to_cartesian(sim_data_dcopy.weights, sim_data_dcopy.verts)

    return sim_data_dcopy

def squash_dimensions(sim_data, dims, val_agg_func):
    assert isinstance(dims, list)

    sim_data_dcopy = copy.deepcopy(sim_data)

    sim_data_dcopy.verts = sim_data_dcopy.verts[:, dims]

    coords_cols = [f'c{d}' for d in dims]
    weights_cols = [f'w{i}' for i in range(sim_data_dcopy.weights.shape[1])]

    df = pd.DataFrame(sim_data_dcopy.coords[:, dims], columns=coords_cols)
    df[weights_cols] = sim_data_dcopy.weights
    df['colors'] = sim_data_dcopy.colors

    df = df.round({col: ROUND_DECIMALS for col in coords_cols})

    df = df.groupby(coords_cols, as_index=False, sort=False).agg({
        **{col: 'mean' for col in weights_cols},
        'colors': val_agg_func
    })

    df = df.round({'colors': ROUND_DECIMALS})
    df['colors'] += 0.0

    sim_data_dcopy.coords = df[coords_cols].to_numpy()
    sim_data_dcopy.weights = df[weights_cols].to_numpy()
    sim_data_dcopy.colors = df['colors'].to_numpy()

    return sim_data_dcopy

def perform_radviz(sim_data):
    sim_data_dcopy = copy.deepcopy(sim_data)

    theta = 2 * np.pi / sim_data_dcopy.verts.shape[0]
    angles = np.arange(sim_data_dcopy.verts.shape[0]) * theta

    sim_data_dcopy.verts = np.column_stack((np.cos(angles), np.sin(angles)))
    sim_data_dcopy.coords = to_cartesian(sim_data_dcopy.weights, sim_data_dcopy.verts)

    """
    TODO:
    aggregation
    streamlit-sortable sort_items
    """

    return sim_data_dcopy
