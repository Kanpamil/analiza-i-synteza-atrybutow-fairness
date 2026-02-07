import streamlit as st
import numpy as np
import plotly.graph_objects as go
import itertools
import re 

# Uruchomienie: streamlit run barycentricVisComp.py

# -------------------------------------------------
# Konfiguracja Strony
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="Fairness Simplex Visualization")

colorscale_jet_full = "Jet"

colorscale_jet_half = [
    [0.0, 'rgb(128, 255, 128)'],
    [0.33, 'rgb(255, 255, 0)'],   
    [0.66, 'rgb(255, 0, 0)'],    
    [1.0, 'rgb(128, 0, 0)']      
]
# -------------------------------------------------
# Logika Matematyczna (Geometria Sympleksu)
# -------------------------------------------------
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 1.0])
C = np.array([1.0, 1.0, 0.0])
D = np.array([0.0, 1.0, 1.0])
STD_TETRA_VERTS = [A, B, C, D]

VAR_KEYS = ['TP_p', 'FP_p', 'TN_p', 'FN_p', 'TP_u', 'FP_u', 'TN_u', 'FN_u']
VAR_LABELS = ['TP (Prot)', 'FP (Prot)', 'TN (Prot)', 'FN (Prot)', 
              'TP (Unprot)', 'FP (Unprot)', 'TN (Unprot)', 'FN (Unprot)']

@st.cache_data
def get_simplex_vertices(n):
    if n == 1: return np.array([[0.0, 0.0, 0.0]])
    elif n == 2: return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elif n == 3: return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3)/2, 0.0]])
    elif n == 4: return np.array(STD_TETRA_VERTS)
    return np.array([])

def f_point(weights, vertices):
    pos = np.zeros(3)
    for w, v in zip(weights, vertices):
        pos += w * v
    return pos

@st.cache_data
def generate_simplex_grid(n, N_res):
    coords, weights_list = [], []
    if n == 0: return np.array([]), np.array([])
    if n == 1: return np.array([[0.0, 0.0, 0.0]]), np.array([[1.0]])
    vertices = get_simplex_vertices(n)
    if n == 2:
        for i in range(N_res + 1):
            w = np.array([i, N_res - i]) / N_res
            coords.append(f_point(w, vertices)); weights_list.append(w)
    elif n == 3:
        for i in range(N_res + 1):
            for j in range(N_res + 1 - i):
                k = N_res - i - j
                w = np.array([i, j, k]) / N_res
                coords.append(f_point(w, vertices)); weights_list.append(w)
    elif n == 4:
        for i in range(N_res + 1):
            for j in range(N_res + 1 - i):
                for k in range(N_res + 1 - i - j):
                    l = N_res - i - j - k
                    w = np.array([i, j, k, l]) / N_res
                    coords.append(f_point(w, vertices)); weights_list.append(w)
    return np.array(coords), np.array(weights_list).T

def get_barycentric_for_slice(x, y, z):
    w_sum = (x + y + z) / 2.0
    return 1.0 - w_sum, w_sum - y, w_sum - z, w_sum - x

# --- Helper dla Metryk Performance ---
def get_performance_metric(tp, fp, tn, fn, metric_name):
    p, n = tp + fn, fp + tn
    total, pred_pos = p + n, tp + fp
    def safe_div(num, den):
        return np.divide(num, den, out=np.full_like(den, np.nan, dtype=float), where=den!=0)
    if metric_name == 'Accuracy': return safe_div(tp + tn, total)
    elif metric_name == 'Precision': return safe_div(tp, pred_pos)
    elif metric_name == 'Recall': return safe_div(tp, p)
    elif metric_name == 'F1 Score':
        pr, rc = safe_div(tp, pred_pos), safe_div(tp, p)
        return safe_div(2 * pr * rc, pr + rc)
    elif metric_name == 'Specificity': return safe_div(tn, n)
    elif metric_name == 'MCC':
        num = (tp * tn) - (fp * fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return safe_div(num, den)
    return np.zeros_like(tp, dtype=float)

# -------------------------------------------------
# Główny Kalkulator Metryk
# -------------------------------------------------
def calculate_metric(inputs, mode):
    def get(k): return inputs[k]
    def sd(n, d): return np.divide(n, d, out=np.full_like(d, np.nan, dtype=float), where=d!=0)
    # 1. Tryb Porównania (Rekurencyjny)
    if mode.startswith('Comparison'):
        match = re.match(r'Comparison:\s*(.*?)\s*\((.+?)\s+vs\s+(.+)\)', mode)
        if match:
            comp_type, f1_mode, f2_mode = match.groups()
            val1 = calculate_metric(inputs, f1_mode.strip())
            val2 = calculate_metric(inputs, f2_mode.strip())
            if comp_type == 'Absolute Difference': return np.abs(val1 - val2)
            elif comp_type == 'Ratio (A/B)': 
                return np.divide(val1, val2, out=np.full_like(val2, np.nan, dtype=float), where=val2!=0)
            elif comp_type == 'Squared Difference': return (val1 - val2)**2
            elif comp_type == 'Minimum': return np.minimum(val1, val2)
        return np.zeros_like(get('TP_p'), dtype=float)
    if mode == "Impossibility_Index":
        # 1. Equal Opportunity (TPR)
        tpr_p = sd(get('TP_p'), get('TP_p')+get('FN_p'))
        tpr_u = sd(get('TP_u'), get('TP_u')+get('FN_u'))
        # 2. Predictive Equality (FPR)
        fpr_p = sd(get('FP_p'), get('FP_p')+get('TN_p'))
        fpr_u = sd(get('FP_u'), get('FP_u')+get('TN_u'))
        # 3. Predictive Parity (PPV / Precision)
        ppv_p = sd(get('TP_p'), get('TP_p')+get('FP_p'))
        ppv_u = sd(get('TP_u'), get('TP_u')+get('FP_u'))
        
        return np.abs(tpr_p - tpr_u) + np.abs(fpr_p - fpr_u) + np.abs(ppv_p - ppv_u)
    # 2. Metryki Fairness (Rozpoznawane po nazwach)
    fairness_keywords = ['Equal Opportunity', 'Predictive Equality', 'Equalized Odds', 
                         'Demographic Parity', 'Disparate Impact', 'Predictive Parity']
    if any(m in mode for m in fairness_keywords):
        tpr_p, fpr_p = sd(get('TP_p'), get('TP_p')+get('FN_p')), sd(get('FP_p'), get('FP_p')+get('TN_p'))
        tpr_u, fpr_u = sd(get('TP_u'), get('TP_u')+get('FN_u')), sd(get('FP_u'), get('FP_u')+get('TN_u'))
        pr_p = sd(get('TP_p')+get('FP_p'), get('TP_p')+get('FN_p')+get('FP_p')+get('TN_p'))
        pr_u = sd(get('TP_u')+get('FP_u'), get('TP_u')+get('FN_u')+get('FP_u')+get('TN_u'))
        ppv_p, ppv_u = sd(get('TP_p'), get('TP_p')+get('FP_p')), sd(get('TP_u'), get('TP_u')+get('FP_u'))

        if 'Equal Opportunity Diff' in mode: return tpr_p - tpr_u
        elif 'Equal Opportunity Ratio' in mode: return sd(tpr_p, tpr_u)
        elif 'Predictive Equality Diff' in mode: return fpr_p - fpr_u
        elif 'Demographic Parity Diff' in mode: return pr_p - pr_u
        elif 'Disparate Impact' in mode: return sd(pr_p, pr_u)
        elif 'Predictive Parity Diff' in mode: return ppv_p - ppv_u

    # 3. Metryki Performance
    try:
        metric_name, scope_part = mode.rsplit(' (', 1)
        scope = scope_part[:-1]
        if scope == 'Prot': return get_performance_metric(get('TP_p'), get('FP_p'), get('TN_p'), get('FN_p'), metric_name)
        elif scope == 'Unprot': return get_performance_metric(get('TP_u'), get('FP_u'), get('TN_u'), get('FN_u'), metric_name)
        elif scope == 'Total': return get_performance_metric(get('TP_p')+get('TP_u'), get('FP_p')+get('FP_u'), get('TN_p')+get('TN_u'), get('FN_p')+get('FN_u'), metric_name)
    except: pass
    return np.zeros_like(get('TP_p'), dtype=float)

# -------------------------------------------------
# Interfejs Streamlit
# -------------------------------------------------
with st.sidebar:
    st.header("Zmienne Macierzy Pomyłek")
    current_values, selected_indices = {}, []
    for i, (key, label) in enumerate(zip(VAR_KEYS, VAR_LABELS)):
        c1, c2 = st.columns([0.15, 0.85])
        if c1.checkbox(label, key=f"chk_{key}", label_visibility="collapsed"):
            selected_indices.append(i); current_values[key] = None
        else:
            current_values[key] = c2.number_input(label, 0, 100, 40 if 'TP' in key or 'TN' in key else 10, key=f"num_{key}")

    n_dim = len(selected_indices)
    slice_pos, slice_axis = 0.5, 'Z'
    if n_dim == 4:
        st.markdown("---")
        slice_axis = st.radio("Oś Cięcia", ["Z", "Y", "X"])
        slice_pos = st.slider("Pozycja Płaszczyzny", 0.0, 1.0, 0.5)

    st.markdown("---")
    st.header("Ustawienia Wizualizacji")
    m_type = st.radio("Tryb", ["Fairness", "Performance", "Comparison","Theorem Visualization"], horizontal=True)
    
    PERF_LIST = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'MCC']
    FAIR_LIST = ['Equal Opportunity Diff (TPR)', 'Equal Opportunity Ratio (TPR)', 'Predictive Equality Diff (FPR)', 
                 'Demographic Parity Diff', 'Disparate Impact (Ratio)', 'Predictive Parity Diff (Precision)']
    SCOPE_MAP = {'Protected (Prot)': 'Prot', 'Unprotected (Unprot)': 'Unprot', 'Total (P+U)': 'Total'}

    if m_type == "Fairness":
        metric_mode = st.selectbox("Wybierz metrykę", FAIR_LIST)
    elif m_type == "Performance":
        m = st.selectbox("Metryka", PERF_LIST); s = st.selectbox("Grupa", list(SCOPE_MAP.keys()))
        metric_mode = f"{m} ({SCOPE_MAP[s]})"
    elif m_type == "Theorem Visualization":
        st.subheader("Test Niemożliwości (Triple Conflict)")
        st.info("Wizualizuje sumę błędów dla: Equal Opportunity, Predictive Equality i Predictive Parity.")
        metric_mode = "Impossibility_Index"
    else:
        comp = st.selectbox("Operacja", ["Absolute Difference", "Ratio (A/B)", "Squared Difference", "Minimum"])
        ALL_OPTS = PERF_LIST + FAIR_LIST
        def sub_ui(lab):
            st.subheader(f"Funkcja {lab}")
            sel_m = st.selectbox(f"Metryka {lab}", ALL_OPTS, key=f'm{lab}')
            if sel_m in FAIR_LIST: return sel_m
            sel_s = st.selectbox(f"Grupa {lab}", list(SCOPE_MAP.keys()), key=f's{lab}')
            return f"{sel_m} ({SCOPE_MAP[sel_s]})"
        metric_mode = f"Comparison: {comp} ({sub_ui('A')} vs {sub_ui('B')})"

    use_log, log_exp = False, 2.0
    if 'Ratio' in metric_mode:
        use_log = st.checkbox("Skala Logarytmiczna")
        if use_log: log_exp = st.number_input("Zakres (10^x)", 0.1, 5.0, 2.0)

    res_3d = st.slider("Rozdzielczość", 5, 100, 40)
    alpha_val = st.slider("Przezroczystość", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.header("Ustawienia Kamery (do PNG)")
    cam_x = st.slider("Kamera X", -3.0, 3.0, 1.5)
    cam_y = st.slider("Kamera Y", -3.0, 3.0, 1.5)
    cam_z = st.slider("Kamera Z", -3.0, 3.0, 1.5)

# -------------------------------------------------
# Obliczenia i Wykresy
# -------------------------------------------------
bary_coords, bary_weights = generate_simplex_grid(n_dim, res_3d)
metric_inputs = {}
w_idx = 0
for key in VAR_KEYS:
    if current_values[key] is None:
        metric_inputs[key] = bary_weights[w_idx] * 100.0; w_idx += 1
    else: metric_inputs[key] = float(current_values[key])

vals_3d = calculate_metric(metric_inputs, metric_mode) if n_dim > 0 else np.array([])
if np.ndim(vals_3d) == 0 and len(bary_coords) > 0: vals_3d = np.full(len(bary_coords), vals_3d)

cmin, cmax = 0.0, 1.0
vals_plot = np.nan_to_num(vals_3d, nan=np.nan)

if 'Diff' in metric_mode or 'MCC' in metric_mode:
    cmin, cmax = -1.0, 1.0
elif 'Ratio' in metric_mode:
    if use_log:
        vals_plot = np.log10(vals_3d)
        cmin, cmax = -log_exp, log_exp
    else:
        valid = vals_3d[np.isfinite(vals_3d) & (vals_3d > 0)]
        cmax = float(np.percentile(valid, 95)) if len(valid) > 0 else 2.0
        cmin = 0.0
current_colorscale = colorscale_jet_full if cmin == -1.0 else colorscale_jet_half

fig_3d = go.Figure()
if n_dim > 0 and len(vals_3d) > 0:
    mask = ~np.isnan(vals_3d)
    fig_3d.add_trace(go.Scatter3d(
        x=bary_coords[mask, 0], y=bary_coords[mask, 1], z=bary_coords[mask, 2],
        mode='markers', marker=dict(size=4, color=vals_plot[mask], colorscale=current_colorscale, 
        cmin=cmin, cmax=cmax, opacity=alpha_val, colorbar=dict(title=metric_mode, x=0))
    ))
    verts = get_simplex_vertices(n_dim)
    for p1, p2 in itertools.combinations(verts, 2):
        fig_3d.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color='black', width=1), showlegend=False))
    for i, idx in enumerate(selected_indices):
        fig_3d.add_trace(go.Scatter3d(x=[verts[i][0]], y=[verts[i][1]], z=[verts[i][2]], mode='text', text=[VAR_KEYS[idx]], textfont=dict(size=12, color="crimson"), showlegend=False))

if n_dim == 4:
    p_coord = np.linspace(0, 1, 10)
    u_p, v_p = np.meshgrid(p_coord, p_coord)
    pos_p = np.full_like(u_p, slice_pos)

    if slice_axis == 'X':
        px, py, pz = pos_p, u_p, v_p
    elif slice_axis == 'Y':
        px, py, pz = u_p, pos_p, v_p
    else:
        px, py, pz = u_p, v_p, pos_p

    fig_3d.add_trace(go.Surface(
        x=px, y=py, z=pz,
        opacity=0.3,          
        showscale=False,      
        colorscale=[[0, 'rgba(255, 0, 0, 0.5)'], [1, 'rgba(255, 0, 0, 0.5)']], 
        name=f'Przekrój {slice_axis}={slice_pos}'
    ))

fig_3d.update_layout(height=700, margin=dict(l=0,r=0,b=0,t=40), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,camera=dict(
            eye=dict(x=cam_x, y=cam_y, z=cam_z)
        )))

col_l, col_r = st.columns([0.5, 0.5])
with col_l: st.plotly_chart(fig_3d, use_container_width=True)
with col_r:
    if n_dim == 4:
        N_H = 80; u = np.linspace(0, 1, N_H); U, V = np.meshgrid(u, u)
        if slice_axis == 'Z': Xh, Yh, Zh = U, V, np.full_like(U, slice_pos)
        elif slice_axis == 'Y': Xh, Zh, Yh = U, V, np.full_like(U, slice_pos)
        else: Yh, Zh, Xh = U, V, np.full_like(U, slice_pos)
        
        w1, w2, w3, w4 = get_barycentric_for_slice(Xh, Yh, Zh)
        mask_2d = (w1>=0) & (w2>=0) & (w3>=0) & (w4>=0)
        
        m_in_2d = current_values.copy()
        for i, w in enumerate([w1, w2, w3, w4]): m_in_2d[VAR_KEYS[selected_indices[i]]] = w * 100.0
        
        v2d = calculate_metric(m_in_2d, metric_mode)
        v2d_p = np.log10(v2d) if 'Ratio' in metric_mode and use_log else v2d
        v2d_p = np.where(mask_2d, v2d_p, np.nan)
        
        fig_2d = go.Figure(go.Heatmap(z=v2d_p, x=u, y=u, colorscale=current_colorscale, zmin=cmin, zmax=cmax))
        fig_2d.update_layout(width=500, height=500, title=f"Przekrój 2D (Oś {slice_axis}={slice_pos})")
        st.plotly_chart(fig_2d)
        
    else:
        st.info("Wybierz 4 zmienne, aby odblokować przekrój 2D.")