import streamlit as st
import numpy as np
from ui.display import grid_3d
from ui.input import metric_selection
from ui.layout import column_comparison
from processing.dimensionality import slice_dimensions
from processing.simulation import SimulationFactory
from processing.constants import ROUND_DECIMALS

def slice():
    column_comparison(visualization)

def visualization(id_):
    is_classification, metric_name, settings = metric_selection(id_)
    sim_data = SimulationFactory.create_simulation(metric_name, st.session_state.resolution)
    sim_data.weights = np.round(sim_data.weights, decimals=ROUND_DECIMALS)

    if is_classification:
        grid_3d(id_, sim_data)
        return

    with settings:
        constraints_key = f'constraints{id_}'

        constraints = st.multiselect(
            'Constraints',
            sim_data.labels,
            max_selections=4,
            on_change=sort_constraints,
            args=(constraints_key, sim_data.labels),
            key=constraints_key
        )

        constraint_to_value = {}

        with st.container(horizontal=True):
            for constraint in constraints:
                index = sim_data.labels.index(constraint)
                unique_values = sorted(set(w[index] for w in sim_data.weights))

                constraint_to_value[constraint] = st.select_slider(
                    constraint,
                    unique_values,
                    key=f'{constraint}_to_value{id_}'
                )

    if len(constraints) == 4:
        sliced_sim_data = slice_dimensions(sim_data, constraint_to_value)
        grid_3d(id_, sliced_sim_data)

def sort_constraints(key, labels):
    if key in st.session_state:
        st.session_state[key] = sorted(st.session_state[key], key=lambda x: labels.index(x))
