import streamlit as st
from ui.display import grid_3d
from ui.input import metric_selection
from ui.layout import column_comparison
from processing.dimensionality import squash_dimensions
from processing.simulation import SimulationFactory

def squash():
    column_comparison(visualization)

def visualization(id_):
    is_classification, metric_name, settings = metric_selection(id_)
    sim_data = SimulationFactory.create_simulation(metric_name, st.session_state.resolution)

    if is_classification:
        grid_3d(id_, sim_data)
        return

    with settings:
        st.info('Barycentric coordinates are aggregated by the mean and values are aggregated according to the '
                'selected function. Every value aggregation function (except "size") ignores NaN values. This means '
                'that if only some of the values being aggregated are NaN, the result is computed from the non-NaN '
                'values and if all of them are NaN, the result is also NaN. The "size" function counts all values, '
                'including NaNs.')

        dims_key = f'dims{id_}'

        dims = st.multiselect(
            'Dimensions',
            range(1, 8),
            max_selections=3,
            on_change=sort_dims,
            args=(dims_key,),
            key=dims_key
        )

        val_agg_func = st.selectbox(
            'Value aggregation function',
            ['Mean', 'Median', 'Min', 'Max', 'Sum', 'Size']
        )

    if len(dims) == 3:
        squashed_sim_data = squash_dimensions(sim_data, [d - 1 for d in dims], val_agg_func.lower())
        grid_3d(id_, squashed_sim_data, agg_labels=True)

def sort_dims(key):
    if key in st.session_state:
        st.session_state[key] = sorted(st.session_state[key])
