import streamlit as st
from streamlit_sortables import sort_items
from core.metrics import FairnessMetrics
from processing.simulation import SimulationData
from ui.input import sidebar_value_agg_func
from ui.metric_mapping import to_display_name, to_metric_name
from ui.plot import plot_simulation_2d

def radviz_projection():
    display_name = st.sidebar.selectbox('Metric', [to_display_name(m) for m in FairnessMetrics.metric_names()])
    metric_name = to_metric_name(display_name)

    sim_data = SimulationData.from_metric_name(metric_name, st.session_state.resolution)

    component_names = sim_data.metrics_class.component_names().tolist()

    with st.sidebar:
        st.write('Drag to reorder components')
        sorted_component_names = sort_items(component_names)

    indices = [component_names.index(name) for name in sorted_component_names]
    projected_sim_data = sim_data.project_radviz(indices, sidebar_value_agg_func().lower())
    plot_simulation_2d(projected_sim_data)
