import streamlit as st
from core.metrics import ClassificationMetrics
from processing.simulation import SimulationData
from ui.metric_mapping import to_display_name, to_metric_name
from ui.plot import plot_simulation_3d

def classification_metric():
    display_name = st.sidebar.selectbox('Metric', [to_display_name(m) for m in ClassificationMetrics.metric_names()])
    metric_name = to_metric_name(display_name)

    sim_data = SimulationData.from_metric_name(metric_name, st.session_state.resolution)
    plot_simulation_3d(sim_data)
