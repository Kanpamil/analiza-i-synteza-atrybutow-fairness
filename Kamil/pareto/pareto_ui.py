import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# --- KONFIGURACJA ŚCIEŻEK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTY KLAS ---
from common.metrics import FairReport, ClassReport
from FairDataGenerator import FairReportGenerator
from logic.data_processor import DataProcessor
from logic.pareto_service import ParetoService

# --- HELPERY ---
def calculate_grid_points(res):
    points_per_group = math.comb(res + 3, 3)
    return points_per_group ** 2

# ==========================================
# 0. KONFIGURACJA NAZW (SŁOWNIK)
# ==========================================
METRIC_PRETTY_NAMES = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'f1_score': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall (TPR)',
    'specificity': 'Specificity (TNR)',
    'mcc': 'MCC',
    'g_mean': 'G-Mean',
    'fnr': 'False Negative Rate',
    'fpr': 'False Positive Rate',
    'tnr': 'True Negative Rate',
    'tpr': 'True Positive Rate',
    'statistical_parity_difference': 'Statistical Parity Difference',
    'statistical_parity_ratio': 'Statistical Parity Ratio',
    'equal_opportunity_difference': 'Equal Opportunity Difference',
    'equal_opportunity_ratio': 'Equal Opportunity Ratio',
    'equalized_odds_difference_avg': 'Equalized Odds Difference (Avg)',
    'equalized_odds_difference_max': 'Equalized Odds Difference (Max)',
    'equalized_odds_ratio_avg': 'Equalized Odds Ratio (Avg)',
    'equalized_odds_ratio_max': 'Equalized Odds Ratio (Max)',
    'predictive_equality_difference': 'Predictive Equality Difference',
    'predictive_equality_ratio': 'Predictive Equality Ratio'
}

def get_pretty_name(metric_key):
    return METRIC_PRETTY_NAMES.get(metric_key, metric_key)

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Fairness Pareto Explorer", layout="wide")

# --- CSS: ZMNIEJSZENIE SIDEBARA ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px;
        max-width: 250px;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Eksplorator Frontu Pareto: Quality vs Fairness")

# Dummy obiekty do pobrania list dostępnych metryk
_dummy_c = ClassReport(0, 0, 0, 0)
_dummy_f = FairReport(_dummy_c, _dummy_c)
QUALITY_METRICS = _dummy_c.available_metrics
FAIRNESS_METRICS = _dummy_f.available_metrics

# ==========================================
# 1. SIDEBAR: PARAMETRY SYMULACJI
# ==========================================
with st.sidebar:
    st.header("⚙️ Ustawienia")
    
    st.subheader("1. Populacja")
    n_total = st.number_input(
        "Populacja (N)",
        min_value=100, max_value=1000000, value=2000, step=100
    )
    p_ratio = st.slider(
        "Proporcja (Prot)", 
        0.05, 0.95, 0.50, 0.05
    )

    st.subheader("2. Base Rates")
    base_rate_p = st.slider("BR (Prot)", 0.01, 0.99, 0.30, 0.01)
    base_rate_u = st.slider("BR (Unp)", 0.01, 0.99, 0.30, 0.01)

    st.divider()
    
    st.subheader("3. Generator")
    mode = st.radio(
        "Tryb", 
        ["Siatka", "Losowy"]
    )
    
    if "Losowy" in mode:
        n_samples = st.number_input(
            "Liczba modeli", 
            min_value=50, max_value=100000, value=2000, step=500
        )
        res_val = 0 
        jitter_val = 0.0
    else:
        # Tryb Siatki
        res_val = st.slider(
            "Rozdzielczość (res)", 
            min_value=2, max_value=15, value=8
        )
        
        jitter_val = st.slider(
            "Jitter", 0.0, 0.05, 0.01, 0.001
        )
        n_samples = None
        
    st.subheader("4. Jakość (Szum)")
    quality_p = st.slider("Jakość P (Youden)", 0.0, 1.0, 1.0, 0.01)
    quality_u = st.slider("Jakość U (Youden)", 0.0, 1.0, 1.0, 0.01)
    
    st.divider()
    st.header("Metryki")
    
    x_metric = st.selectbox(
        "Oś X", 
        QUALITY_METRICS, 
        index=QUALITY_METRICS.index('accuracy') if 'accuracy' in QUALITY_METRICS else 0,
        format_func=get_pretty_name
    )
    y_metric = st.selectbox(
        "Oś Y", 
        FAIRNESS_METRICS, 
        index=FAIRNESS_METRICS.index('statistical_parity_difference') if 'statistical_parity_difference' in FAIRNESS_METRICS else 0,
        format_func=get_pretty_name
    )

    # --- NOWOŚĆ: Metoda Ograniczeń ---
    st.divider()
    st.header("⚖️ Ograniczenia")
    constraint_val = st.slider(
        f"Min. {get_pretty_name(y_metric)}", 
        min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )

# ==========================================
# 2. LOGIKA BIZNESOWA
# ==========================================

gen = FairReportGenerator(
    n_total=n_total, 
    p_ratio=p_ratio, 
    base_rate_p=base_rate_p, 
    base_rate_u=base_rate_u
)

with st.spinner('Generowanie...'):
    if "Losowy" in mode:
        report = gen.generate_random(n_samples=n_samples)
    else:
        try:
            report = gen.generate_from_simplex(res=res_val, max_samples=None, jitter=jitter_val)
        except TypeError:
            report = gen.generate_from_simplex(res=res_val, max_samples=None)

processor = DataProcessor(report)
df = processor.prepare_dataframe(x_metric, y_metric)

tpr_p_real = df['TP_prot'] / (df['TP_prot'] + df['FN_prot'] + 1e-9)
tpr_u_real = df['TP_unp'] / (df['TP_unp'] + df['FN_unp'] + 1e-9)
fpr_p_real = df['FP_prot'] / (df['FP_prot'] + df['TN_prot'] + 1e-9)
fpr_u_real = df['FP_unp'] / (df['FP_unp'] + df['TN_unp'] + 1e-9)
youden_p = tpr_p_real - fpr_p_real
youden_u = tpr_u_real - fpr_u_real

quality_mask = (youden_p <= quality_p) & (youden_u <= quality_u)
df = df[quality_mask]

possible_cols = [c for c in df.columns if c not in [x_metric, 'is_pareto', 'Type'] 
                 and not c.startswith(('norm_', 'raw_', 'TP_', 'FP_', 'TN_', 'FN_'))]
current_y_col = possible_cols[0] 

if not df.empty:
    cols_to_clip = [x_metric, current_y_col]
    df[cols_to_clip] = df[cols_to_clip].clip(lower=0.0, upper=1.0)

if not df.empty:
    df['is_pareto'] = ParetoService.identify_pareto(
        df, 
        x_col=f'norm_{x_metric}', 
        y_col=f'norm_{current_y_col}'
    )
    df['Type'] = df['is_pareto'].map({True: 'Front Pareto', False: 'Punkty zdominowane'})
else:
    st.warning("Brak punktów w wybranym zakresie.")
    st.stop()

# 1. Metoda Utopii (Istniejąca)
df['dist_to_utopia'] = np.sqrt(
    (1 - df[x_metric])**2 + 
    (1 - df[current_y_col])**2
)
best_idx = df['dist_to_utopia'].idxmin()
best_point = df.loc[best_idx]

# 2. Metoda Knee Point (NOWOŚĆ: Obliczenia)
try:
    knee_idx = ParetoService.get_knee_point_index(
        df, 
        x_col=x_metric, 
        y_col=current_y_col, 
        pareto_mask=df['is_pareto']
    )
    knee_point = df.loc[knee_idx]
except AttributeError:
    # Zabezpieczenie gdyby ParetoService nie był zaktualizowany
    knee_idx = best_idx
    knee_point = best_point

# ==========================================
# 3. WIZUALIZACJA (PLOTLY)
# ==========================================

col1, col2 = st.columns([3, 1])

with col1:
    COLOR_PARETO = "#104E8B"
    COLOR_DOMINATED = "rgba(160, 160, 160, 0.3)"
    COLOR_AXIS = "#333333"            
    COLOR_GRID = "#F0F0F0"            

    # --- KONTROLKI ---
    c_check, c_radio = st.columns([1, 2])
    with c_check:
        show_dominated = st.checkbox("Pokaż punkty zdominowane", value=True)
    with c_radio:
        color_mode = st.radio(
            "Tryb kolorowania:",
            ["Front Pareto (Klasyczny)", "Mapa Ciepła (Dystans)"],
            index=1, 
            horizontal=True,
            label_visibility="collapsed"
        )
    st.write("---")

    cols_for_hover = [
        'TP_prot', 'FP_prot', 'TN_prot', 'FN_prot',
        'TP_unp', 'FP_unp', 'TN_unp', 'FN_unp',
        'Type', 'dist_to_utopia'
    ]

    # --- FILTROWANIE ---
    df_viz = df.copy()
    if not show_dominated:
        df_viz = df_viz[df_viz['is_pareto'] == True]

    # --- SORTOWANIE ---
    if color_mode == "Mapa Ciepła (Dystans)":
        df_viz = df_viz.sort_values(by='dist_to_utopia', ascending=False)
    else:
        df_viz = df_viz.sort_values(by='is_pareto', ascending=True)

    # --- RYSOWANIE PUNKTÓW ---
    if color_mode == "Front Pareto (Klasyczny)":
        fig = px.scatter(
            df_viz, 
            x=x_metric, 
            y=current_y_col,
            color='Type', 
            template="plotly_white",
            color_discrete_map={
                'Front Pareto': COLOR_PARETO,
                'Punkty zdominowane': COLOR_DOMINATED
            },
            hover_data=cols_for_hover,
            render_mode='webgl'
        )
    else:
        fig = px.scatter(
            df_viz, 
            x=x_metric, 
            y=current_y_col,
            color='dist_to_utopia', 
            template="plotly_white",
            color_continuous_scale='RdYlBu', 
            range_color=[0, 1.42],
            hover_data=cols_for_hover,
            render_mode='webgl'
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Dystans do (1,1)",
                thickness=100,
                thicknessmode="pixels",
                len=0.8,
                yanchor="middle",
                y=0.5,
                xpad=10
            )
        )

    # --- HOVER TEMPLATE ---
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

    # --- LINIA FRONTU (Zawsze czarna) ---
    pareto_points = df[df['is_pareto']].sort_values(by=x_metric)
    
    fig.add_trace(go.Scattergl(
        x=pareto_points[x_metric], 
        y=pareto_points[current_y_col],
        mode='lines', 
        name='Linia Frontu', 
        line=dict(color="#000000", width=4),
        hoverinfo='skip'
    ))
    
    if len(pareto_points) >= 2 and color_mode == "Mapa Ciepła (Dystans)":
        # Sortowanie mamy już wyżej (by=x_metric), więc iloc[0] i iloc[-1] to skrajne punkty X
        p_start = pareto_points.iloc[0]
        p_end = pareto_points.iloc[-1]
        
        fig.add_trace(go.Scattergl(
            x=[p_start[x_metric], p_end[x_metric]],
            y=[p_start[current_y_col], p_end[current_y_col]],
            mode='lines',
            name='Linia bazowa (Knee)',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dash'),
            hoverinfo='text',
            hovertext="Linia odniesienia dla Knee Point"
        ))

    # --- PUNKT IDEALNY ---
    fig.add_trace(go.Scattergl(
        x=[1.0],
        y=[1.0],
        mode='markers',
        name='Punkt Idealny',
        marker=dict(
            color='#FFD700',
            size=15,
            symbol='diamond',
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext="<b>PUNKT IDEALNY</b><br>(1.0, 1.0)"
    ))

    # --- PUNKT OPTYMALNY (Metoda Utopii) ---
    if color_mode == "Mapa Ciepła (Dystans)":
        fig.add_trace(go.Scattergl(
            x=[best_point[x_metric]],
            y=[best_point[current_y_col]],
            mode='markers',
            name='Najlepszy Kompromis',
            marker=dict(
                color='#00CC96',
                size=18,
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=(
                f"<b>🏆 NAJLEPSZY KOMPROMIS</b><br>"
                f"Dystans: {best_point['dist_to_utopia']:.4f}<br>"
                f"X: {best_point[x_metric]:.4f}<br>"
                f"Y: {best_point[current_y_col]:.4f}"
            )
        ))

    # --- KNEE POINT (NOWOŚĆ: Wizualizacja) ---
    # Rysujemy fioletowy romb (tylko w trybie mapy ciepła dla czytelności, lub zawsze)
    # Jeśli knee point jest różny od best point
    if color_mode == "Mapa Ciepła (Dystans)":
        fig.add_trace(go.Scattergl(
            x=[knee_point[x_metric]],
            y=[knee_point[current_y_col]],
            mode='markers',
            name='Knee Point',
            marker=dict(
                color='#AB63FA',
                size=16,
                symbol='diamond',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext="<b>Knee Point</b><br>Punkt Przegięcia"
        ))

    # --- LINIA OGRANICZENIA (NOWOŚĆ) ---
    if color_mode == "Mapa Ciepła (Dystans)":
        fig.add_hline(
            y=constraint_val, 
            line_dash="dash", 
            line_color="#2CA02C", 
        )

    # --- LAYOUT ---
    fig.update_layout(
        width=1200,   
        height=900,
        xaxis_title=get_pretty_name(x_metric), 
        yaxis_title=f"Fairness (1 - {get_pretty_name(y_metric)})" if "difference" in y_metric else get_pretty_name(y_metric),
        plot_bgcolor='white',
        font=dict(family="Arial", size=18, color="black"),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            range=[-0.02, 1.05], dtick=0.1, showline=True, linewidth=1, 
            linecolor=COLOR_AXIS, mirror=True, showgrid=True, gridcolor=COLOR_GRID, constrain='domain'
        ),
        yaxis=dict(
            range=[-0.02, 1.05], dtick=0.1, showline=True, linewidth=1, 
            linecolor=COLOR_AXIS, mirror=True, showgrid=True, gridcolor=COLOR_GRID, scaleanchor="x", scaleratio=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key="main_pareto_chart")

with col2:
    st.subheader("Statystyki")
    st.metric("Modele", f"{len(df):,}")
    st.metric("Pareto", f"{len(pareto_points):,}")
    
    st.write("---")
    st.markdown("### Kompromis (Utopia)")
    st.write(f"**Dystans:** {best_point['dist_to_utopia']:.4f}")
    st.write(f"**{get_pretty_name(x_metric)}:** {best_point[x_metric]:.4f}")
    st.write(f"**Fairness:** {best_point[current_y_col]:.4f}")
    
    # Dodanie statystyk Knee Point jeśli jest inny
    if knee_idx != best_idx:
        st.write("---")
        st.markdown("### Knee Point")
        st.write(f"**{get_pretty_name(x_metric)}:** {knee_point[x_metric]:.4f}")
        st.write(f"**Fairness:** {knee_point[current_y_col]:.4f}")

    st.write("---")
    if st.button("Resetuj"):
        st.rerun()