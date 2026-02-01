# plots.py
import plotly.express as px
import plotly.graph_objects as go
from app_config import get_pretty_name  # Import helpera

def create_pareto_chart(df, x_metric, y_metric, current_y_col, 
                        best_point, knee_point, constraint_val, 
                        color_mode, show_dominated):
    
    COLOR_PARETO = "#104E8B"
    COLOR_DOMINATED = "rgba(160, 160, 160, 0.3)"
    COLOR_AXIS = "#333333"            
    COLOR_GRID = "#F0F0F0" 

    df_viz = df.copy()
    if not show_dominated:
        df_viz = df_viz[df_viz['is_pareto'] == True]

    if color_mode == "Mapa Ciepła (Dystans)":
        df_viz = df_viz.sort_values(by='dist_to_utopia', ascending=False)
    else:
        df_viz = df_viz.sort_values(by='is_pareto', ascending=True)

    cols_for_hover = [
        'TP_prot', 'FP_prot', 'TN_prot', 'FN_prot',
        'TP_unp', 'FP_unp', 'TN_unp', 'FN_unp',
        'Type', 'dist_to_utopia'
    ]

    if color_mode == "Front Pareto (Klasyczny)":
        fig = px.scatter(
            df_viz, 
            x=x_metric, y=current_y_col,
            color='Type', 
            template="plotly_white",
            color_discrete_map={'Front Pareto': COLOR_PARETO, 'Punkty zdominowane': COLOR_DOMINATED},
            hover_data=cols_for_hover,
            render_mode='webgl'
        )
    else:
        fig = px.scatter(
            df_viz, 
            x=x_metric, y=current_y_col,
            color='dist_to_utopia', 
            template="plotly_white",
            color_continuous_scale='RdYlBu', 
            range_color=[0, 1.42],
            hover_data=cols_for_hover,
            render_mode='webgl'
        )
        fig.update_layout(coloraxis_colorbar=dict(
            title="Dystans do (1,1)", thickness=15, len=0.5, yanchor="middle", y=0.5
        ))

    my_hover_template = (
        "<b>%{customdata[8]}</b><br>" + 
        "---<br>" +
        f"<b>{get_pretty_name(x_metric)}:</b> %{{x:.4f}}<br>" +
        f"<b>{get_pretty_name(y_metric)}:</b> %{{y:.4f}}<br>" +
        "<b>Dystans:</b> %{customdata[9]:.4f}<br>" +
        "---<br>" +
        "<b>Grupa Chroniona (P):</b><br>" +
        "TP: %{customdata[0]} | FN: %{customdata[3]}<br>" +
        "FP: %{customdata[1]} | TN: %{customdata[2]}<br>" +
        "---<br>" +
        "<b>Grupa Niechroniona (U):</b><br>" +
        "TP: %{customdata[4]} | FN: %{customdata[7]}<br>" +
        "FP: %{customdata[5]} | TN: %{customdata[6]}<br>" +
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=my_hover_template)

    pareto_points = df[df['is_pareto']].sort_values(by=x_metric)
    fig.add_trace(go.Scattergl(
        x=pareto_points[x_metric], y=pareto_points[current_y_col],
        mode='lines', name='Linia Frontu', line=dict(color="#000000", width=4), hoverinfo='skip'
    ))

    if len(pareto_points) >= 2 and color_mode == "Mapa Ciepła (Dystans)":
        p_start, p_end = pareto_points.iloc[0], pareto_points.iloc[-1]
        fig.add_trace(go.Scattergl(
            x=[p_start[x_metric], p_end[x_metric]], y=[p_start[current_y_col], p_end[current_y_col]],
            mode='lines', name='Linia bazowa (Knee)',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dash')
        ))

    fig.add_trace(go.Scattergl(
        x=[1.0], y=[1.0], mode='markers', name='Punkt Idealny',
        marker=dict(color='#FFD700', size=15, symbol='diamond', line=dict(width=2, color='black'))
    ))
    
    if color_mode == "Mapa Ciepła (Dystans)":
        fig.add_trace(go.Scattergl(
            x=[best_point[x_metric]], y=[best_point[current_y_col]],
            mode='markers', name='Najlepszy Kompromis',
            marker=dict(color='#00CC96', size=18, symbol='star', line=dict(width=2, color='black'))
        ))
        fig.add_trace(go.Scattergl(
            x=[knee_point[x_metric]], y=[knee_point[current_y_col]],
            mode='markers', name='Knee Point',
            marker=dict(color='#AB63FA', size=16, symbol='diamond', line=dict(width=2, color='black'))
        ))
        fig.add_hline(y=constraint_val, line_dash="dash", line_color="#2CA02C")

    fig.update_layout(
            width=900,
            height=900,
            xaxis_title=get_pretty_name(x_metric),
            yaxis_title=f"Fairness (1 - {get_pretty_name(y_metric)})" if "difference" in y_metric else get_pretty_name(y_metric),
            plot_bgcolor='white',
            font=dict(family="Arial", size=18, color="black"),
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            
            xaxis=dict(
                range=[-0.02, 1.05], 
                dtick=0.1, 
                showline=True, 
                linewidth=1, 
                linecolor=COLOR_AXIS, 
                mirror=True, 
                showgrid=True, 
                gridcolor=COLOR_GRID, 
                constrain='domain'
            ),
            
            yaxis=dict(
                range=[-0.02, 1.05], 
                dtick=0.1, 
                showline=True, 
                linewidth=1, 
                linecolor=COLOR_AXIS, 
                mirror=True, 
                showgrid=True, 
                gridcolor=COLOR_GRID, 
                
                scaleanchor="x",
                scaleratio=1
            )
        )
    return fig
    