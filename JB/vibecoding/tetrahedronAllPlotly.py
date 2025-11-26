import streamlit as st
import numpy as np
import plotly.graph_objects as go
import itertools

# Run with: streamlit run JB/vibecoding/tetrahedronAllPlotly.py

# -------------------------------------------------
#  Konfiguracja Strony
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="Fairness Simplex Visualization")

# -------------------------------------------------
#  Logika Matematyczna (Numpy)
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
    coords = []
    weights_list = []
    
    if n == 0: return np.array([]), np.array([])
    if n == 1: return np.array([[0.0, 0.0, 0.0]]), np.array([[1.0]])

    vertices = get_simplex_vertices(n)

    if n == 2:
        for i in range(N_res + 1):
            w = np.array([i, N_res - i]) / N_res
            coords.append(f_point(w, vertices))
            weights_list.append(w)
    elif n == 3:
        for i in range(N_res + 1):
            for j in range(N_res + 1 - i):
                k = N_res - i - j
                w = np.array([i, j, k]) / N_res
                coords.append(f_point(w, vertices))
                weights_list.append(w)
    elif n == 4:
        for i in range(N_res + 1):
            for j in range(N_res + 1 - i):
                for k in range(N_res + 1 - i - j):
                    l = N_res - i - j - k
                    w = np.array([i, j, k, l]) / N_res
                    coords.append(f_point(w, vertices))
                    weights_list.append(w)

    return np.array(coords), np.array(weights_list).T

def get_barycentric_for_slice(x, y, z):
    w_sum = (x + y + z) / 2.0
    w1 = 1.0 - w_sum
    w2 = w_sum - y
    w3 = w_sum - z
    w4 = w_sum - x
    return w1, w2, w3, w4

# --- Helper for Performance Metrics ---
def get_performance_metric(tp, fp, tn, fn, metric_name):
    p = tp + fn
    n = fp + tn
    total = p + n
    pred_pos = tp + fp
    
    def safe_div(num, den):
        return np.divide(num, den, out=np.full_like(den, np.nan, dtype=float), where=den!=0)

    if metric_name == 'Accuracy':
        return safe_div(tp + tn, total)
    elif metric_name == 'Precision':
        return safe_div(tp, pred_pos)
    elif metric_name == 'Recall':
        return safe_div(tp, p)
    elif metric_name == 'F1 Score':
        prec = safe_div(tp, pred_pos)
        rec = safe_div(tp, p)
        return safe_div(2 * prec * rec, prec + rec)
    elif metric_name == 'Specificity':
        return safe_div(tn, n)
    elif metric_name == 'MCC':
        num = (tp * tn) - (fp * fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return safe_div(num, den)
        
    return np.zeros_like(tp, dtype=float)

# -------------------------------------------------
#  Główny Kalkulator Metryk
# -------------------------------------------------
def calculate_metric(inputs, mode):
    def get(k): return inputs[k]
    
    # >>>> FAIRNESS METRICS <<<<
    if 'Equal Opportunity' in mode or 'Predictive Equality' in mode or \
       'Equalized Odds' in mode or 'Demographic Parity' in mode or \
       'Disparate Impact' in mode or 'Predictive Parity Diff' in mode:
       
        tp_p, fn_p = get('TP_p'), get('FN_p')
        fp_p, tn_p = get('FP_p'), get('TN_p')
        p_p = tp_p + fn_p
        n_p = fp_p + tn_p
        
        tp_u, fn_u = get('TP_u'), get('FN_u')
        fp_u, tn_u = get('FP_u'), get('TN_u')
        p_u = tp_u + fn_u
        n_u = fp_u + tn_u
        
        def safe_div(n, d): return np.divide(n, d, out=np.full_like(d, np.nan, dtype=float), where=d!=0)

        tpr_p = safe_div(tp_p, p_p)
        fpr_p = safe_div(fp_p, n_p)
        pr_p  = safe_div(tp_p + fp_p, p_p + n_p)
        ppv_p = safe_div(tp_p, tp_p + fp_p)

        tpr_u = safe_div(tp_u, p_u)
        fpr_u = safe_div(fp_u, n_u)
        pr_u  = safe_div(tp_u + fp_u, p_u + n_u)
        ppv_u = safe_div(tp_u, tp_u + fp_u)

        if mode == 'Equal Opportunity Diff (TPR)': return tpr_p - tpr_u
        elif mode == 'Equal Opportunity Ratio (TPR)': return safe_div(tpr_p, tpr_u)
        elif mode == 'Predictive Equality Diff (FPR)': return fpr_p - fpr_u
        elif mode == 'Equalized Odds Diff (Avg)': return 0.5 * (np.abs(tpr_p - tpr_u) + np.abs(fpr_p - fpr_u))
        elif mode == 'Equalized Odds (Max Diff)': return np.maximum(np.abs(tpr_p - tpr_u), np.abs(fpr_p - fpr_u))
        elif mode == 'Demographic Parity Diff': return pr_p - pr_u
        elif mode == 'Disparate Impact (Ratio)': return safe_div(pr_p, pr_u)
        elif mode == 'Predictive Parity Diff (Precision)': return ppv_p - ppv_u

    # >>>> PERFORMANCE METRICS <<<<
    else:
        try:
            metric_name, scope_part = mode.rsplit(' (', 1)
            scope = scope_part[:-1]
        except ValueError:
            return np.zeros_like(get('TP_p'), dtype=float)

        if scope == 'Prot':
            return get_performance_metric(get('TP_p'), get('FP_p'), get('TN_p'), get('FN_p'), metric_name)
        elif scope == 'Unprot':
            return get_performance_metric(get('TP_u'), get('FP_u'), get('TN_u'), get('FN_u'), metric_name)
        elif scope == 'Total':
            return get_performance_metric(get('TP_p')+get('TP_u'), 
                                          get('FP_p')+get('FP_u'), 
                                          get('TN_p')+get('TN_u'), 
                                          get('FN_p')+get('FN_u'), metric_name)

    return np.zeros_like(get('TP_p'), dtype=float)

# -------------------------------------------------
#  Interfejs Streamlit
# -------------------------------------------------

with st.sidebar:
    st.header("Zmienne Macierzy Pomyłek")
    st.write("Zaznacz do 4 zmiennych.")
    
    if 'selected_vars' not in st.session_state:
        st.session_state.selected_vars = []
    
    current_values = {}
    selected_indices = []
    
    for i, (key, label) in enumerate(zip(VAR_KEYS, VAR_LABELS)):
        col1, col2 = st.columns([0.15, 0.85])
        
        is_selected = col1.checkbox(label, key=f"chk_{key}", label_visibility="collapsed")
        
        if is_selected:
            col2.markdown(f"**{label}** (Zakres 0-100)")
            selected_indices.append(i)
            current_values[key] = None 
        else:
            val = col2.number_input(label, min_value=0, max_value=100, value=40 if 'TP' in key or 'TN' in key else 10, step=1, key=f"num_{key}")
            current_values[key] = val

    n_dim = len(selected_indices)
    if n_dim > 4:
        st.error("Zaznaczono więcej niż 4 zmienne! Odznacz nadmiarowe.")
        st.stop()

    # --- SEKCJA 1: Ustawienia Przekroju (RESTORED) ---
    # Te zmienne muszą być zainicjalizowane przed głównym panelem
    slice_pos = 0.5
    slice_axis = 'Z' 
    
    if n_dim == 4:
        st.markdown("---")
        st.header("Ustawienia Przekroju 2D")
        
        var_names = [VAR_KEYS[idx] for idx in selected_indices]
        label_z = f"Z ({var_names[1]}-{var_names[3]} vs {var_names[0]}-{var_names[2]})"
        label_y = f"Y ({var_names[1]}-{var_names[2]} vs {var_names[0]}-{var_names[3]})"
        label_x = f"X ({var_names[2]}-{var_names[3]} vs {var_names[0]}-{var_names[1]})"
        
        slice_options = [label_z, label_y, label_x]
        slice_axis_str = st.radio("Oś Cięcia", slice_options)
        
        # Mapowanie stringa z powrotem na kod osi
        if slice_axis_str == label_z: slice_axis = 'Z'
        elif slice_axis_str == label_y: slice_axis = 'Y'
        else: slice_axis = 'X'
            
        slice_pos = st.slider("Pozycja Płaszczyzny", 0.0, 1.0, 0.5)

    # --- SEKCJA 2: Ustawienia Wizualizacji ---
    st.markdown("---")
    st.header("Ustawienia Wizualizacji")
    
    metric_type = st.radio("Typ Metryki", ["Fairness (P vs U)", "Performance (Trafność)"], horizontal=True)
    
    metric_mode = ""
    
    if metric_type == "Fairness (P vs U)":
        metric_options = [
            'Equal Opportunity Diff (TPR)', 
            'Equal Opportunity Ratio (TPR)', 
            'Predictive Equality Diff (FPR)', 
            'Equalized Odds Diff (Avg)',
            'Equalized Odds (Max Diff)',
            'Demographic Parity Diff', 
            'Disparate Impact (Ratio)',
            'Predictive Parity Diff (Precision)'
        ]
        metric_mode = st.selectbox("Wybierz Metrykę Fairness", metric_options)
        
    else:
        col_perf1, col_perf2 = st.columns(2)
        
        perf_metric = col_perf1.selectbox("Metryka", [
            'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'MCC'
        ])
        
        perf_scope_label = col_perf2.selectbox("Grupa", [
            'Protected (Prot)', 'Unprotected (Unprot)', 'Total (P+U)'
        ])
        
        scope_map = {
            'Protected (Prot)': 'Prot',
            'Unprotected (Unprot)': 'Unprot',
            'Total (P+U)': 'Total'
        }
        metric_mode = f"{perf_metric} ({scope_map[perf_scope_label]})"

    use_log = False
    log_exp = 2.0
    if 'Ratio' in metric_mode:
        col_log1, col_log2 = st.columns([0.5, 0.5])
        use_log = col_log1.checkbox("Skala Log", value=False)
        if use_log:
            log_exp = col_log2.number_input("Wykładnik (10^x)", value=2.0, min_value=0.1, max_value=5.0)
            
    res_3d = st.slider("Rozdzielczość Punktów 3D", 5, 100, 40)
    alpha_val = st.slider("Przeźroczystość punktów (Alpha)", 0.0, 1.0, 0.5, step=0.05)

# -------------------------------------------------
#  Główny Panel
# -------------------------------------------------

bary_coords, bary_weights = generate_simplex_grid(n_dim, res_3d)

metric_inputs_3d = {}
w_idx = 0
for key in VAR_KEYS:
    if current_values[key] is None:
        if w_idx < bary_weights.shape[0]:
            metric_inputs_3d[key] = bary_weights[w_idx] * 100.0
            w_idx += 1
        else:
            metric_inputs_3d[key] = np.array([])
    else:
        metric_inputs_3d[key] = float(current_values[key])

if n_dim > 0:
    vals_3d = calculate_metric(metric_inputs_3d, metric_mode)
    
    # --- FIX: Obsługa skalara (TypeError: has no len) ---
    # Jeśli wynik to pojedyncza liczba (skalar), rozciągnij go na wszystkie punkty
    if np.ndim(vals_3d) == 0:
        if bary_coords.shape[0] > 0:
            vals_3d = np.full(bary_coords.shape[0], vals_3d)
        else:
            vals_3d = np.array([])
            
    vals_3d = np.nan_to_num(vals_3d, nan=np.nan, posinf=np.nan, neginf=np.nan)
else:
    vals_3d = np.array([])

# --- Zakresy Kolorów ---
cmin, cmax = 0.0, 1.0
colorscale = 'Jet'

if 'Diff' in metric_mode or 'MCC' in metric_mode:
    cmin, cmax = -1.0, 1.0
elif 'Ratio' in metric_mode:
    if use_log:
        cmin, cmax = -log_exp, log_exp
        vals_3d_plot = np.log10(vals_3d)
    else:
        cmin, cmax = 0.0, 2.0
        vals_3d_plot = vals_3d
elif metric_mode in ['Accuracy (Prot)', 'Recall (Prot)', 'Precision (Prot)', 'F1 Score (Prot)', 'Specificity (Prot)']:
    cmin, cmax = 0.0, 1.0
else:
    vals_3d_plot = vals_3d
    
if 'Ratio' not in metric_mode:
    vals_3d_plot = vals_3d

if "Fairness" in metric_type: # Używamy metric_type bo metric_mode jest różne
    # Możemy spróbować wyciągnąć info
    if 'Diff' in metric_mode: suffix = "[Diff]"
    elif 'Ratio' in metric_mode: suffix = "[Ratio]"
    else: suffix = ""
    legend_title = f"{metric_mode}"
else:
    legend_title = f"{metric_mode}"

# --- WYKRES 3D ---
fig_3d = go.Figure()

if n_dim > 0 and len(vals_3d) > 0:
    mask_valid = ~np.isnan(vals_3d)
    
    if np.any(mask_valid):
        fig_3d.add_trace(go.Scatter3d(
            x=bary_coords[mask_valid, 0],
            y=bary_coords[mask_valid, 1],
            z=bary_coords[mask_valid, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=vals_3d_plot[mask_valid],
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                opacity=alpha_val,
                colorbar=dict(title=legend_title, x=0)
            ),
            name='Valid',
            hovertemplate='Val: %{marker.color:.2f}<extra></extra>'
        ))

    if np.any(~mask_valid):
        fig_3d.add_trace(go.Scatter3d(
            x=bary_coords[~mask_valid, 0],
            y=bary_coords[~mask_valid, 1],
            z=bary_coords[~mask_valid, 2],
            mode='markers',
            marker=dict(size=5, color='magenta', opacity=alpha_val),
            name='NaN/Error'
        ))

if n_dim > 0:
    verts = get_simplex_vertices(n_dim)
    if n_dim > 1:
        for p1, p2 in itertools.combinations(verts, 2):
            fig_3d.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color='black', width=2), showlegend=False
            ))
    
    for i, idx in enumerate(selected_indices):
        v_pos = verts[i]
        label_text = VAR_KEYS[idx]
        fig_3d.add_trace(go.Scatter3d(
            x=[v_pos[0]], y=[v_pos[1]], z=[v_pos[2]],
            mode='text',
            text=[label_text],
            textposition="top center",
            textfont=dict(size=14, color="crimson", family="Arial Black"),
            showlegend=False
        ))

# --- PŁASZCZYZNA 2D i SURFACE ---
fig_2d = None
stats_title = ""

# Definicja pustych zmiennych, żeby nie było błędu przy n_dim < 4
surf_x, surf_y, surf_z = [], [], []

if n_dim == 4:
    N_SURF_LOW = 25 
    u_low = np.linspace(0, 1, N_SURF_LOW)
    v_low = np.linspace(0, 1, N_SURF_LOW)
    U_low, V_low = np.meshgrid(u_low, v_low)
    
    N_HEATMAP = 100
    u_high = np.linspace(0, 1, N_HEATMAP)
    v_high = np.linspace(0, 1, N_HEATMAP)
    U_high, V_high = np.meshgrid(u_high, v_high)
    
    if slice_axis == 'Z':
        X_s, Y_s, Z_s = U_low, V_low, np.full_like(U_low, slice_pos)
        X_h, Y_h, Z_h = U_high, V_high, np.full_like(U_high, slice_pos)
        lbl_x, lbl_y = VAR_KEYS[selected_indices[1]], VAR_KEYS[selected_indices[2]]
    elif slice_axis == 'Y':
        X_s, Z_s, Y_s = U_low, V_low, np.full_like(U_low, slice_pos)
        X_h, Z_h, Y_h = U_high, V_high, np.full_like(U_high, slice_pos)
        lbl_x, lbl_y = VAR_KEYS[selected_indices[1]], VAR_KEYS[selected_indices[3]]
    else:
        Y_s, Z_s, X_s = U_low, V_low, np.full_like(U_low, slice_pos)
        Y_h, Z_h, X_h = U_high, V_high, np.full_like(U_high, slice_pos)
        lbl_x, lbl_y = VAR_KEYS[selected_indices[2]], VAR_KEYS[selected_indices[3]]

    w1s, w2s, w3s, w4s = get_barycentric_for_slice(X_s, Y_s, Z_s)
    margin = 0.1 
    mask_inside_s = (w1s>=-margin) & (w2s>=-margin) & (w3s>=-margin) & (w4s>=-margin)
    
    # Dane dla Surface 3D
    surf_x = np.where(mask_inside_s, X_s, np.nan)
    surf_y = np.where(mask_inside_s, Y_s, np.nan)
    surf_z = np.where(mask_inside_s, Z_s, np.nan)

    w1h, w2h, w3h, w4h = get_barycentric_for_slice(X_h, Y_h, Z_h)
    eps = 1e-9
    mask_inside_h = (w1h>=-eps) & (w2h>=-eps) & (w3h>=-eps) & (w4h>=-eps)
    
    metric_inputs_2d = current_values.copy()
    metric_inputs_2d[VAR_KEYS[selected_indices[0]]] = w1h * 100.0
    metric_inputs_2d[VAR_KEYS[selected_indices[1]]] = w2h * 100.0
    metric_inputs_2d[VAR_KEYS[selected_indices[2]]] = w3h * 100.0
    metric_inputs_2d[VAR_KEYS[selected_indices[3]]] = w4h * 100.0
    
    vals_2d = calculate_metric(metric_inputs_2d, metric_mode)
    
    # FIX: Obsługa skalara dla 2D
    if np.ndim(vals_2d) == 0:
        vals_2d = np.full((N_HEATMAP, N_HEATMAP), vals_2d)
        
    vals_2d = np.nan_to_num(vals_2d, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    valid_vals = vals_2d[mask_inside_h]
    stats_title = "Statystyki Przekroju"
    if len(valid_vals) > 0 and np.any(~np.isnan(valid_vals)):
        m = np.nanmean(valid_vals)
        s = np.nanstd(valid_vals)
        stats_title = f"Przekrój: Mean={m:.4f}, Std={s:.4f}"
    
    vals_2d_masked = np.where(mask_inside_h, vals_2d, np.nan)
    
    if 'Ratio' in metric_mode and use_log:
        vals_2d_plot = np.log10(vals_2d_masked)
    else:
        vals_2d_plot = vals_2d_masked

    fig_2d = go.Figure(data=go.Heatmap(
        z=vals_2d_plot,
        x=u_high,
        y=v_high,
        colorscale=colorscale,
        zmin=cmin,
        zmax=cmax,
        colorbar=dict(title=legend_title)
    ))
    
    fig_2d.update_layout(
        xaxis_title=lbl_x,
        yaxis_title=lbl_y,
        width=700, height=700,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )

# Dodanie śladu Surface do 3D
if n_dim == 4:
    fig_3d.add_trace(go.Surface(
        x=surf_x, y=surf_y, z=surf_z,
        opacity=0.3, showscale=False, colorscale='Reds',
        name='Slice Plane'
    ))
else:
    fig_3d.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False))

# --- Layout 3D ---
fig_3d.update_layout(
    title=f"Simplex Dimension: {max(0, n_dim-1)} ({n_dim} Zmiennych)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
    uirevision='constant' 
)

col_main, col_2d = st.columns([0.5, 0.5])

with col_main:
    st.plotly_chart(fig_3d, use_container_width=True, key="3d_plot")

with col_2d:
    if n_dim == 4 and fig_2d:
        st.subheader(stats_title)
        st.plotly_chart(fig_2d) 
    elif n_dim < 4:
        st.info("Wybierz dokładnie 4 zmienne, aby zobaczyć przekrój 2D.")
    else:
        st.empty()