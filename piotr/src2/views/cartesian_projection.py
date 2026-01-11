import streamlit as st
from core.metrics import FairnessMetrics
from processing.simulation import SimulationData
from ui.input import sidebar_value_agg_func
from ui.metric_mapping import to_display_name, to_metric_name
from ui.plot import plot_simulation_3d

REQUIRED_DIMENSIONS = 3

def cartesian_projection():
    display_name = st.sidebar.selectbox('Metric', [to_display_name(m) for m in FairnessMetrics.metric_names()])
    metric_name = to_metric_name(display_name)

    sim_data = SimulationData.from_metric_name(metric_name, st.session_state.resolution)

    key = 'dimensions'

    dimensions = st.sidebar.multiselect(
        'Dimensions',
        range(1, 8),
        max_selections=3,
        key=key,
        on_change=_sort_dimensions,
        args=(key,)
    )

    value_agg_func = sidebar_value_agg_func()

    if len(dimensions) == REQUIRED_DIMENSIONS:
        indices = [d - 1 for d in dimensions]
        projected_sim_data = sim_data.project_cartesian(indices, value_agg_func.lower())
        plot_simulation_3d(projected_sim_data)
    else:
        missing = REQUIRED_DIMENSIONS - len(dimensions)
        st.info(f'Please select {missing} more dimension{'s' if missing > 1 else ''} to generate a visualization')

def _sort_dimensions(key):
    if key in st.session_state:
        st.session_state[key] = sorted(st.session_state[key])
