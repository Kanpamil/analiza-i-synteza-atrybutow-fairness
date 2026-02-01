import os
import sys
import streamlit as st
import numpy as np

# Konfiguracja ścieżek
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from common.metrics import FairReport, ClassReport
from logic.fair_data_generator import FairReportGenerator
from logic.data_processor import DataProcessor
from logic.pareto_service import ParetoService
import app_config as cfg
import plots

st.set_page_config(page_title="Fairness Pareto Explorer", layout="wide")
st.markdown(cfg.SIDEBAR_CSS, unsafe_allow_html=True)
st.title("Eksplorator Frontu Pareto: Quality vs Fairness")

_dummy_c = ClassReport(0, 0, 0, 0)
_dummy_f = FairReport(_dummy_c, _dummy_c)
QUALITY_METRICS = _dummy_c.available_metrics
FAIRNESS_METRICS = _dummy_f.available_metrics

with st.sidebar:
    st.header("Ustawienia")
    
    st.subheader("1. Populacja")
    n_total = st.number_input("Populacja (N)", 100, 1000000, 2000, 100)
    p_ratio = st.slider("Proporcja (Prot)", 0.05, 0.95, 0.50, 0.05)

    st.subheader("2. Base Rates")
    base_rate_p = st.slider("BR (Prot)", 0.01, 0.99, 0.30, 0.01)
    base_rate_u = st.slider("BR (Unp)", 0.01, 0.99, 0.30, 0.01)

    st.divider()
    st.subheader("3. Generator")
    mode = st.radio("Tryb", ["Siatka", "Losowy"])
    
    if "Losowy" in mode:
        n_samples = st.number_input("Liczba modeli", 50, 100000, 2000, 500)
        res_val, jitter_val = 0, 0.0
    else:
        res_val = st.slider("Rozdzielczość (res)", 2, 15, 8)
        jitter_val = st.slider("Jitter", 0.0, 0.05, 0.01, 0.001)
        n_samples = None
        
    st.subheader("4. Jakość (Szum)")
    quality_p = st.slider("Jakość P (Youden)", 0.0, 1.0, 1.0, 0.01)
    quality_u = st.slider("Jakość U (Youden)", 0.0, 1.0, 1.0, 0.01)
    
    st.divider()
    st.header("Metryki")
    x_metric = st.selectbox("Oś X", QUALITY_METRICS, index=0, format_func=cfg.get_pretty_name)
    y_metric = st.selectbox("Oś Y", FAIRNESS_METRICS, index=0, format_func=cfg.get_pretty_name)

    st.divider()
    st.header("⚖️ Ograniczenia")
    constraint_val = st.slider(f"Min. {cfg.get_pretty_name(y_metric)}", 0.0, 1.0, 0.8, 0.05)

gen = FairReportGenerator(n_total, p_ratio, base_rate_p, base_rate_u)

with st.spinner('Generowanie...'):
    if "Losowy" in mode:
        report = gen.generate_random(n_samples=n_samples)
    else:
        report = gen.generate_from_simplex(res=res_val, jitter=jitter_val)

processor = DataProcessor(report)
df = processor.prepare_dataframe(x_metric, y_metric)

tpr_p = df['TP_prot'] / (df['TP_prot'] + df['FN_prot'] + 1e-9)
tpr_u = df['TP_unp'] / (df['TP_unp'] + df['FN_unp'] + 1e-9)
fpr_p = df['FP_prot'] / (df['FP_prot'] + df['TN_prot'] + 1e-9)
fpr_u = df['FP_unp'] / (df['FP_unp'] + df['TN_unp'] + 1e-9)
df = df[(tpr_p - fpr_p <= quality_p) & (tpr_u - fpr_u <= quality_u)]

if df.empty:
    st.warning("Brak punktów w wybranym zakresie.")
    st.stop()

possible_cols = [c for c in df.columns if c not in [x_metric] and not c.startswith(('norm_', 'raw_', 'TP_', 'FP_', 'TN_', 'FN_'))]
current_y_col = possible_cols[0]
df[[x_metric, current_y_col]] = df[[x_metric, current_y_col]].clip(0.0, 1.0)

df['is_pareto'] = ParetoService.identify_pareto(df, f'norm_{x_metric}', f'norm_{current_y_col}')
df['Type'] = df['is_pareto'].map({True: 'Front Pareto', False: 'Punkty zdominowane'})
df['dist_to_utopia'] = np.sqrt((1 - df[x_metric])**2 + (1 - df[current_y_col])**2)

best_idx = df['dist_to_utopia'].idxmin()
best_point = df.loc[best_idx]
knee_idx = ParetoService.get_knee_point_index(df, x_metric, current_y_col, df['is_pareto'])
knee_point = df.loc[knee_idx]

col1, col2 = st.columns([3, 1])

with col1:
    c_check, c_radio = st.columns([1, 2])
    show_dominated = c_check.checkbox("Pokaż punkty zdominowane", value=True)
    color_mode = c_radio.radio("Tryb kolorowania:", ["Front Pareto (Klasyczny)", "Mapa Ciepła (Dystans)"], index=1, horizontal=True, label_visibility="collapsed")
    
    st.write("---")
    
    fig = plots.create_pareto_chart(
        df, x_metric, y_metric, current_y_col, 
        best_point, knee_point, constraint_val, 
        color_mode, show_dominated
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Statystyki")
    st.metric("Modele", f"{len(df):,}")
    st.metric("Pareto", f"{df['is_pareto'].sum():,}")
    
    st.write("---")
    st.markdown("### Kompromis (Utopia)")
    st.write(f"**Dystans:** {best_point['dist_to_utopia']:.4f}")
    st.write(f"**{cfg.get_pretty_name(x_metric)}:** {best_point[x_metric]:.4f}")
    st.write(f"**Fairness:** {best_point[current_y_col]:.4f}")
    
    if knee_idx != best_idx:
        st.write("---")
        st.markdown("### Knee Point")
        st.write(f"**{cfg.get_pretty_name(x_metric)}:** {knee_point[x_metric]:.4f}")
        st.write(f"**Fairness:** {knee_point[current_y_col]:.4f}")

    st.write("---")
    if st.button("Resetuj"):
        st.rerun()