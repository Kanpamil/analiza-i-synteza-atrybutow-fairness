import streamlit as st
import plotly.graph_objects as go
from processing.utils import get_hovers, get_aggregated_verts_labels, get_shifted_verts, get_edges

FONT_SIZE_SMALL = 14
FONT_SIZE_BIG = 18
EDGE_WIDTH = 1

def grid_2d(id_, sim_data):
    assert sim_data.verts.shape[1] == 2
    assert sim_data.verts.shape[0] == len(sim_data.labels)

    assert sim_data.coords.shape[1] == 2
    assert sim_data.coords.shape[0] == sim_data.weights.shape[0] == sim_data.colors.shape[0]

    data = [go.Scattergl(
        x=sim_data.coords[:, 0],
        y=sim_data.coords[:, 1],
        mode='markers',
        marker=dict(
            size=st.session_state.point_size,
            opacity=st.session_state.point_opacity,
            color=sim_data.colors,
            colorscale='Jet',
            showscale=True
        ),
        text=get_hovers(sim_data),
        hoverinfo='text',
        hoverlabel=dict(font=dict(size=FONT_SIZE_SMALL)),
        showlegend=False
    )]

    if st.session_state.show_edges:
        edges = get_edges(sim_data)

        data.append(go.Scattergl(
            x=edges[:, 0],
            y=edges[:, 1],
            mode='lines',
            line=dict(color='white', width=EDGE_WIDTH),
            hoverinfo='none',
            showlegend=False
        ))

    sim_data.verts = get_shifted_verts(sim_data)

    data.append(go.Scatter(
        x=sim_data.verts[:, 0],
        y=sim_data.verts[:, 1],
        mode='text',
        text=sim_data.labels,
        textfont=dict(size=FONT_SIZE_BIG),
        textposition='middle center',
        hoverinfo='none',
        showlegend=False
    ))

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )

    fig = go.Figure(data, layout)
    st.plotly_chart(fig, height=st.session_state.chart_height, key=f'fig{id_}')

def grid_3d(id_, sim_data, agg_labels=False):
    assert sim_data.verts.shape[1] == 3
    assert sim_data.verts.shape[0] == len(sim_data.labels)

    assert sim_data.coords.shape[1] == 3
    assert sim_data.coords.shape[0] == sim_data.weights.shape[0] == sim_data.colors.shape[0]

    data = [go.Scatter3d(
        x=sim_data.coords[:, 0],
        y=sim_data.coords[:, 1],
        z=sim_data.coords[:, 2],
        mode='markers',
        marker=dict(
            size=st.session_state.point_size,
            opacity=st.session_state.point_opacity,
            color=sim_data.colors,
            colorscale='Jet',
            showscale=True
        ),
        text=get_hovers(sim_data),
        hoverinfo='text',
        hoverlabel=dict(font=dict(size=FONT_SIZE_SMALL)),
        showlegend=False
    )]

    if st.session_state.show_edges:
        edges = get_edges(sim_data)

        data.append(go.Scatter3d(
            x=edges[:, 0],
            y=edges[:, 1],
            z=edges[:, 2],
            mode='lines',
            line=dict(color='white', width=EDGE_WIDTH),
            hoverinfo='none',
            showlegend=False
        ))

    if agg_labels:
        sim_data.verts, sim_data.labels = get_aggregated_verts_labels(sim_data)

    sim_data.verts = get_shifted_verts(sim_data)

    data.append(go.Scatter3d(
        x=sim_data.verts[:, 0],
        y=sim_data.verts[:, 1],
        z=sim_data.verts[:, 2],
        mode='text',
        text=sim_data.labels,
        textfont=dict(size=FONT_SIZE_BIG),
        textposition='middle center',
        hoverinfo='none',
        showlegend=False
    ))

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data, layout)
    st.plotly_chart(fig, height=st.session_state.chart_height, key=f'fig{id_}')
