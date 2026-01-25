import streamlit as st
import plotly.io as pio
import plotly.colors as pc
import views

st.set_page_config(layout='wide')

pg = st.navigation([
        st.Page(views.barycentric_slicing, title='Barycentric slicing'),
        st.Page(views.cartesian_projection, title='Cartesian projection'),
        st.Page(views.radviz_projection, title='Radviz projection'),
        st.Page(views.classification_metric, title='Classification metric')
    ],
    position='top'
)

st.sidebar.header('Settings')
st.sidebar.divider()
st.session_state.chart_width = st.sidebar.slider('Chart width', min_value=600, max_value=2600, value=1600, step=100)
st.session_state.chart_height = st.sidebar.slider('Chart height', min_value=400, max_value=1600, value=1000, step=100)
st.session_state.resolution = st.sidebar.slider('Resolution', min_value=1, max_value=20, value=10, step=1)
st.session_state.point_size = st.sidebar.slider('Point size', min_value=1, max_value=20, value=5, step=1)
st.session_state.point_opacity = st.sidebar.slider('Point opacity', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
st.session_state.font_size = st.sidebar.slider('Font size', min_value=10, max_value=30, value=15, step=1)
st.session_state.show_labels = st.sidebar.toggle('Show labels', value=True)
st.session_state.show_wireframe = st.sidebar.toggle('Show wireframe', value=False)
if st.session_state.show_wireframe:
    st.session_state.wireframe_width = st.sidebar.slider('Wireframe width', min_value=1, max_value=10, value=2, step=1)
st.session_state.color_scale = st.sidebar.selectbox('Color scale', ['auto'] + pc.named_colorscales())

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
