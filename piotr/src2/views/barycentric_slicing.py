import streamlit as st
from core.metrics import FairnessMetrics
from processing.simulation import SimulationData
from ui.metric_mapping import to_display_name, to_metric_name
from ui.plot import plot_simulation_3d

REQUIRED_COMPONENTS = 4

def barycentric_slicing():
    display_name = st.sidebar.selectbox('Metric', [to_display_name(m) for m in FairnessMetrics.metric_names()])
    metric_name = to_metric_name(display_name)

    sim_data = SimulationData.from_metric_name(metric_name, st.session_state.resolution)
    rounded_sim_data = sim_data.round_points_barycentric()

    key = 'selected_component_names'
    component_names = rounded_sim_data.metrics_class.component_names().tolist()

    selected_component_names = st.sidebar.multiselect(
        'Components',
        component_names,
        max_selections=4,
        key=key,
        on_change=_sort_selected_component_names,
        args=(key, component_names)
    )

    index_to_value = {}

    for selected_component_name in selected_component_names:
        index = component_names.index(selected_component_name)
        unique_values = sorted(set(b[index] for b in rounded_sim_data.points_barycentric))

        index_to_value[index] = st.sidebar.select_slider(selected_component_name, unique_values)

    if len(index_to_value) == REQUIRED_COMPONENTS:
        sliced_sim_data = rounded_sim_data.slice_barycentric(index_to_value)
        plot_simulation_3d(sliced_sim_data)
    else:
        missing = REQUIRED_COMPONENTS - len(index_to_value)
        st.info(f'Please select {missing} more component{'s' if missing > 1 else ''} to generate a visualization')

def _sort_selected_component_names(key, component_names):
    if key in st.session_state:
        st.session_state[key] = sorted(st.session_state[key], key=lambda x: component_names.index(x))
