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
    x_metric = st.selectbox("Oś X", QUALITY_METRICS, index=QUALITY_METRICS.index('accuracy') if 'accuracy' in QUALITY_METRICS else 0)
    y_metric = st.selectbox("Oś Y", FAIRNESS_METRICS, index=FAIRNESS_METRICS.index('statistical_parity_difference') if 'statistical_parity_difference' in FAIRNESS_METRICS else 0)

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

# Wyliczamy TPR/FPR do Youdena (dla filtrowania jakości)
tpr_p_real = df['TP_prot'] / (df['TP_prot'] + df['FN_prot'] + 1e-9)
tpr_u_real = df['TP_unp'] / (df['TP_unp'] + df['FN_unp'] + 1e-9)
fpr_p_real = df['FP_prot'] / (df['FP_prot'] + df['TN_prot'] + 1e-9)
fpr_u_real = df['FP_unp'] / (df['FP_unp'] + df['TN_unp'] + 1e-9)

youden_p = tpr_p_real - fpr_p_real
youden_u = tpr_u_real - fpr_u_real

# Maska Jakości
quality_mask = (youden_p <= quality_p) & (youden_u <= quality_u)
df = df[quality_mask]

# Wykrywanie nazwy kolumny Y
possible_cols = [c for c in df.columns if c not in [x_metric, 'is_pareto', 'Type'] 
                 and not c.startswith(('norm_', 'raw_', 'TP_', 'FP_', 'TN_', 'FN_'))]
current_y_col = possible_cols[0] 

# Wyznaczanie Frontu Pareto
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

# Obliczanie odległości od ideału
df['dist_to_utopia'] = np.sqrt(
    (1 - df[x_metric])**2 + 
    (1 - df[current_y_col])**2
)

# Znajdowanie punktu optymalnego
best_idx = df['dist_to_utopia'].idxmin()
best_point = df.loc[best_idx]

# ==========================================
# 3. WIZUALIZACJA (PLOTLY)
# ==========================================

col1, col2 = st.columns([3, 1])

with col1:
    COLOR_PARETO = "#104E8B"
    COLOR_DOMINATED = "rgba(160, 160, 160, 0.3)" 
    COLOR_AXIS = "#333333"            
    COLOR_GRID = "#F0F0F0"            

    color_mode = st.radio(
        "Tryb kolorowania:",
        ["Front Pareto (Klasyczny)", "Mapa Ciepła (Dystans)"],
        horizontal=True
    )

    cols_for_hover = [
        'TP_prot', 'FP_prot', 'TN_prot', 'FN_prot',
        'TP_unp', 'FP_unp', 'TN_unp', 'FN_unp',
        'Type', 'dist_to_utopia'
    ]

    # 1. SORTOWANIE
    if color_mode == "Mapa Ciepła (Dystans)":
        df = df.sort_values(by='dist_to_utopia', ascending=False)
    else:
        df = df.sort_values(by='is_pareto', ascending=True)

    # 2. RYSOWANIE PUNKTÓW
    if color_mode == "Front Pareto (Klasyczny)":
        fig = px.scatter(
            df, 
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
            df, 
            x=x_metric, 
            y=current_y_col,
            color='dist_to_utopia', 
            template="plotly_white",
            color_continuous_scale='RdYlBu', 
            range_color=[0, 1.42],
            hover_data=cols_for_hover,
            render_mode='webgl'
        )
        # --- KONFIGURACJA PASKA KOLORÓW (WYŚRODKOWANIE) ---
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Dystans do",
                thickness=100,        # Szerokość paska
                thicknessmode="pixels",
                len=0.8,              # Długość paska: 80% wysokości wykresu
                yanchor="middle",     # Kotwica: środek paska
                y=0.5,                # Pozycja Y: środek osi (0.5)
                xpad=10               # Lekki odstęp od osi wykresu
            )
        )

    my_hover_template = (
        "<b>%{customdata[8]}</b><br>" +
        f"Jakość ({x_metric}): %{{x:.4f}}<br>" +
        f"{current_y_col}: %{{y:.4f}}<br>" +
        "Dystans do (1,1): %{customdata[9]:.4f}<br>" +
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=my_hover_template)

    # 3. LINIA FRONTU
    pareto_points = df[df['is_pareto']].sort_values(by=x_metric)
    line_color = "#000000" if color_mode == "Mapa Ciepła (Dystans)" else COLOR_PARETO
    
    fig.add_trace(go.Scattergl(
        x=pareto_points[x_metric], 
        y=pareto_points[current_y_col],
        mode='lines', 
        name='Linia Frontu', 
        line=dict(color=line_color, width=4),
        hoverinfo='skip'
    ))

    # 4. PUNKT IDEALNY (1,1)
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

    # 5. PUNKT OPTYMALNY
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

    # 6. LAYOUT
    fig.update_layout(
        width=1200,   # Szeroki kontener - gwarantuje miejsce dla paska
        height=900,   # Sztywna wysokość
        
        xaxis_title=x_metric, 
        yaxis_title=current_y_col,
        plot_bgcolor='white',
        
        # Marginesy
        margin=dict(l=50, r=50, t=50, b=50),

        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""
        ),
        
        # OŚ X: 
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
        
        # OŚ Y: Wymuszenie proporcji kwadratowych względem X
        yaxis=dict(
            range=[-0.02, 1.05], 
            dtick=0.1, 
            showline=True, 
            linewidth=1, 
            linecolor=COLOR_AXIS, 
            mirror=True, 
            showgrid=True, 
            gridcolor=COLOR_GRID,
            
            scaleanchor="x",  # <--- Skaluj Y względem X
            scaleratio=1      # <--- 1:1 (Kwadrat)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key="main_pareto_chart")

with col2:
    st.subheader("Statystyki")
    st.metric("Modele", f"{len(df):,}")
    st.metric("Pareto", f"{len(pareto_points):,}")
    
    st.write("---")
    st.markdown("### Kompromis")
    st.write(f"**Dystans:** {best_point['dist_to_utopia']:.4f}")
    st.write(f"**{x_metric}:** {best_point[x_metric]:.4f}")
    st.write(f"**Fairness:** {best_point[current_y_col]:.4f}")

    st.write("---")
    if st.button("Resetuj"):
        st.rerun()