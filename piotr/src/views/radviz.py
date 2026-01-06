import streamlit as st
from ui.display import grid_2d
from ui.input import metric_selection
from ui.layout import column_comparison
from processing.dimensionality import perform_radviz
from processing.simulation import SimulationFactory

def radviz():
    column_comparison(visualization)

def visualization(id_):
    _, metric_name, _ = metric_selection(id_)
    sim_data = SimulationFactory.create_simulation(metric_name, st.session_state.resolution)

    radviz_sim_data = perform_radviz(sim_data)
    grid_2d(id_, radviz_sim_data)
