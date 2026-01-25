import streamlit as st
from streamlit_theme import st_theme
import numpy as np
import warnings
import plotly.graph_objects as go
from processing.utils import create_hovers, create_offset_vertices, create_wireframe_2d, create_wireframe_3d

def plot_simulation_2d(sim_data):
    _plot_simulation(
        sim_data=sim_data,
        trace_func=go.Scattergl,
        text_trace_func=go.Scatter,
        wireframe=create_wireframe_2d(sim_data) if st.session_state.show_wireframe else None,
        layout_additional_dict={'yaxis': dict(scaleanchor='x', scaleratio=1)}
    )

def plot_simulation_3d(sim_data):
    _plot_simulation(
        sim_data=sim_data,
        trace_func=go.Scatter3d,
        text_trace_func=go.Scatter3d,
        wireframe=create_wireframe_3d(sim_data) if st.session_state.show_wireframe else None,
    )

def _plot_simulation(sim_data,
                     trace_func,
                     text_trace_func,
                     wireframe,
                     layout_additional_dict=None):
    if layout_additional_dict is None:
        layout_additional_dict = {}

    data = []
    nan_mask = np.isnan(sim_data.point_values)
    valid_mask = ~nan_mask
    hovers = create_hovers(sim_data)

    if np.any(valid_mask):
        if st.session_state.color_scale == 'auto':
            cmin, cmax, color_scale = _choose_color_scale(sim_data)
        else:
            cmin, cmax, color_scale = -np.inf, np.inf, st.session_state.color_scale

        data.append(trace_func(
            **_create_coordinate_dict(sim_data.points_cartesian[valid_mask]),
            mode='markers',
            marker=dict(
                size=st.session_state.point_size,
                opacity=st.session_state.point_opacity,
                color=sim_data.point_values[valid_mask],
                cmin=cmin,
                cmax=cmax,
                colorscale=color_scale,
                showscale=True
            ),
            text=hovers[valid_mask],
            hoverinfo='text',
            hoverlabel=dict(font=dict(size=st.session_state.font_size)),
            showlegend=False
        ))

    if np.any(nan_mask):
        data.append(trace_func(
            **_create_coordinate_dict(sim_data.points_cartesian[nan_mask]),
            mode='markers',
            marker=dict(
                size=st.session_state.point_size,
                opacity=st.session_state.point_opacity,
                color='magenta',
                showscale=False
            ),
            text=hovers[nan_mask],
            hoverinfo='text',
            hoverlabel=dict(font=dict(size=st.session_state.font_size)),
            showlegend=False
        ))

    if wireframe is not None:
        data.append(trace_func(
            **_create_coordinate_dict(wireframe),
            mode='lines',
            line=dict(color=_choose_wireframe_color(), width=st.session_state.wireframe_width),
            hoverinfo='none',
            showlegend=False
        ))

    if st.session_state.show_labels:
        offset_vertices = create_offset_vertices(sim_data)

        data.append(text_trace_func(
            **_create_coordinate_dict(offset_vertices),
            mode='text',
            text=sim_data.vertex_labels,
            textfont=dict(size=st.session_state.font_size),
            textposition='middle center',
            hoverinfo='none',
            showlegend=False
        ))

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision='constant',
        **layout_additional_dict
    )

    st.plotly_chart(go.Figure(data, layout), width=st.session_state.chart_width, height=st.session_state.chart_height)

def _create_coordinate_dict(array):
    coordinate_names = ['x', 'y', 'z']

    return {
        coordinate_name: array[:, i]
        for i, coordinate_name in enumerate(coordinate_names[:array.shape[1]])
    }

def _choose_color_scale(sim_data):
    color_rules = [
        (-1.0, 1.0, 'jet'),
        (0.0, np.inf, 'solar'),
        (-np.inf, np.inf, 'viridis')
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        min_value = np.nanmin(sim_data.point_values)
        max_value = np.nanmax(sim_data.point_values)

    for cmin, cmax, color_scale in color_rules:
        is_min_ok = min_value >= cmin or np.isclose(min_value, cmin)
        is_max_ok = max_value <= cmax or np.isclose(max_value, cmax)

        if is_min_ok and is_max_ok:
            return cmin, cmax, color_scale

def _choose_wireframe_color():
    theme = st_theme()

    if theme is not None and theme['base'] == 'light':
        return 'black'

    return 'white'
