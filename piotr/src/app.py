import streamlit as st
import plotly.io as pio
import views

st.set_page_config(layout='wide')

pg = st.navigation([
        st.Page(lambda: st.write('placeholder'), title='Home'),
        st.Page(views.slice, title='Slice'),
        st.Page(views.squash, title='Squash'),
        st.Page(views.radviz, title='Radviz')
    ],
    position='top'
)

global_settings_disabled = pg.title in ['Home']

st.sidebar.header('Global settings')
st.session_state.chart_height = st.sidebar.slider(
    'Chart height', min_value=400, max_value=1200, value=900, step=100, disabled=global_settings_disabled)
st.session_state.resolution = st.sidebar.slider(
    'Resolution', min_value=1, max_value=20, value=10, step=1, disabled=global_settings_disabled)
st.session_state.point_size = st.sidebar.slider(
    'Point size', min_value=1, max_value=20, value=5, step=1, disabled=global_settings_disabled)
st.session_state.point_opacity = st.sidebar.slider(
    'Point opacity', min_value=0.0, max_value=1.0, value=1.0, step=0.1, disabled=global_settings_disabled)
st.session_state.is_comparison = st.sidebar.toggle(
    'Comparison', disabled=global_settings_disabled)
st.session_state.show_edges = st.sidebar.toggle(
    'Edges', disabled=global_settings_disabled)

pio.templates['no_axes'] = dict(
    layout=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
)

pio.templates.default = 'plotly+no_axes'

pg.run()
