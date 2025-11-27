import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import itertools

# -------------------------------------------------
#  Konfiguracja Zmiennych
# -------------------------------------------------
VAR_KEYS = ['TP_p', 'FP_p', 'TN_p', 'FN_p', 'TP_u', 'FP_u', 'TN_u', 'FN_u']
VAR_LABELS = ['TP (Prot)', 'FP (Prot)', 'TN (Prot)', 'FN (Prot)', 
              'TP (Unprot)', 'FP (Unprot)', 'TN (Unprot)', 'FN (Unprot)']
# Zmieniamy domyślne wartości na znormalizowane (0-1)
DEFAULT_VALS = [0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1]

vars_state = []
for key, label, val in zip(VAR_KEYS, VAR_LABELS, DEFAULT_VALS):
    vars_state.append({
        'key': key,
        'label': label,
        'value': val,
        'selected': False,
        'slider': None,
        'textbox': None,
        'checkbox': None
    })

# Global settings
N_RES_SIMPLEX = 12  
N_RES_GRID = 100    
bary_coords = np.array([])
bary_weights = np.array([])

# -------------------------------------------------
#  Geometria Simpleksów
# -------------------------------------------------
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 1.0])
C = np.array([1.0, 1.0, 0.0])
D = np.array([0.0, 1.0, 1.0])
STD_TETRA_VERTS = [A, B, C, D]

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

# -------------------------------------------------
#  Metryki
# -------------------------------------------------
def calculate_metric(inputs, mode):
    def get(k): return inputs[k]
    
    # Protected
    p_prot = get('TP_p') + get('FN_p')
    n_prot = get('FP_p') + get('TN_p')
    
    tpr_p = np.divide(get('TP_p'), p_prot, out=np.full_like(p_prot, np.nan, dtype=float), where=p_prot!=0)
    fpr_p = np.divide(get('FP_p'), n_prot, out=np.full_like(n_prot, np.nan, dtype=float), where=n_prot!=0)
    pr_p  = np.divide(get('TP_p') + get('FP_p'), p_prot + n_prot, out=np.full_like(p_prot, np.nan, dtype=float), where=(p_prot+n_prot)!=0)
    prec_p = np.divide(get('TP_p'), get('TP_p') + get('FP_p'), out=np.full_like(p_prot, np.nan, dtype=float), where=(get('TP_p')+get('FP_p'))!=0)

    # Unprotected
    p_unprot = get('TP_u') + get('FN_u')
    n_unprot = get('FP_u') + get('TN_u')
    
    tpr_u = np.divide(get('TP_u'), p_unprot, out=np.full_like(p_unprot, np.nan, dtype=float), where=p_unprot!=0)
    fpr_u = np.divide(get('FP_u'), n_unprot, out=np.full_like(n_unprot, np.nan, dtype=float), where=n_unprot!=0)
    pr_u  = np.divide(get('TP_u') + get('FP_u'), p_unprot + n_unprot, out=np.full_like(p_unprot, np.nan, dtype=float), where=(p_unprot+n_unprot)!=0)
    prec_u = np.divide(get('TP_u'), get('TP_u') + get('FP_u'), out=np.full_like(p_unprot, np.nan, dtype=float), where=(get('TP_u')+get('FP_u'))!=0)

    if mode == 'Equal Opportunity (TPR Diff)': return tpr_p - tpr_u
    elif mode == 'Equal Opportunity (Ratio)': return np.divide(tpr_p, tpr_u, out=np.full_like(tpr_p, np.nan, dtype=float), where=tpr_u!=0)
    elif mode == 'Predictive Equality (FPR Diff)': return fpr_p - fpr_u
    elif mode == 'Equalized Odds (Avg Diff)': return 0.5 * (np.abs(tpr_p - tpr_u) + np.abs(fpr_p - fpr_u))
    elif mode == 'Demographic Parity (PR Diff)': return pr_p - pr_u
    elif mode == 'Predictive Parity (Prec Diff)': return prec_p - prec_u
    elif mode == 'Global Recall':
        tp_tot = get('TP_p') + get('TP_u')
        p_tot = p_prot + p_unprot
        return np.divide(tp_tot, p_tot, out=np.full_like(tp_tot, np.nan, dtype=float), where=p_tot!=0)
        
    return np.zeros_like(p_prot, dtype=float)

# -------------------------------------------------
#  GUI Setup
# -------------------------------------------------
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.25, right=0.95, bottom=0.20, wspace=0.25)

# Zmienne (Lewy panel)
row_height = 0.03
start_y = 0.88
for i, var in enumerate(vars_state):
    y = start_y - i * (row_height + 0.015)
    
    # Checkbox
    ax_c = plt.axes([0.02, y, 0.02, row_height], frameon=False)
    var['checkbox'] = CheckButtons(ax_c, [''], [False])
    
    # Label
    plt.text(0.045, y + row_height/2, var['label'], transform=fig.transFigure, va='center', fontsize=9)
    
    # TextBox (Value) - Przesunięte, żeby zniknięcie było widoczne
    ax_t = plt.axes([0.13, y, 0.04, row_height])
    var['textbox'] = TextBox(ax_t, '', initial=f"{var['value']:.2f}")
    
    # Slider (Value) - Zakres 0-1
    ax_s = plt.axes([0.18, y, 0.05, row_height])
    # Zmiana: Zakres 0.0 do 1.0
    var['slider'] = Slider(ax_s, '', 0.0, 1.0, valinit=var['value'], color='gray')
    var['slider'].valtext.set_visible(False)

# Panel 3D
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_axis_off()

# Panel 2D
ax2d = fig.add_subplot(122)
stats_text = ax2d.text(0.5, -0.1, "", transform=ax2d.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')

# Dolne kontrolki
ax_metric = plt.axes([0.02, 0.05, 0.20, 0.25], facecolor='#f0f0f0')
radio_metric = RadioButtons(ax_metric, [
    'Equal Opportunity (TPR Diff)', 'Equal Opportunity (Ratio)', 
    'Predictive Equality (FPR Diff)', 'Equalized Odds (Avg Diff)',
    'Demographic Parity (PR Diff)', 'Global Recall'
], active=0)

ax_check_log = plt.axes([0.30, 0.08, 0.08, 0.05], frameon=False)
check_log = CheckButtons(ax_check_log, ['Log Scale'], [False])
ax_text_log = plt.axes([0.38, 0.085, 0.03, 0.04])
text_log = TextBox(ax_text_log, 'Exp:', initial="2")

ax_check_trans = plt.axes([0.30, 0.13, 0.08, 0.05], frameon=False)
check_transparency = CheckButtons(ax_check_trans, ['Transp.'], [True])

ax_slider_pos = plt.axes([0.50, 0.10, 0.30, 0.03])
slider_pos = Slider(ax_slider_pos, 'Cut Pos', 0.0, 1.0, valinit=0.5)

ax_res_box = plt.axes([0.85, 0.02, 0.05, 0.04])
text_res = TextBox(ax_res_box, 'Res:', initial=str(N_RES_SIMPLEX))

ax_radio_axis = plt.axes([0.50, 0.02, 0.25, 0.06], frameon=False)
radio_axis = RadioButtons(ax_radio_axis, ['Z (Axis 0 vs 3)', 'Y (Axis 0 vs 2)', 'X (Axis 0 vs 1)'], active=0)

# -------------------------------------------------
#  Stan Wykresów
# -------------------------------------------------
plot_objs = {
    'sc_valid': None,
    'sc_nan': None,
    'surf': None,
    'img': None,
    'cbar': None,
    'grid_2d': None,
    'last_slice_state': {'pos': -1, 'axis': -1}
}

cmap_jet = plt.cm.jet.copy()
cmap_jet.set_bad(color='white')

# -------------------------------------------------
#  Logika Aktualizacji
# -------------------------------------------------

def init_geometry():
    global bary_coords, bary_weights
    
    ax3d.clear()
    ax3d.set_axis_off()
    
    selected_indices = [i for i, v in enumerate(vars_state) if v['selected']]
    n = len(selected_indices)
    
    ax3d.set_title(f"Simplex Dimension: {max(0, n-1)} ({n} Vars)", fontsize=10)
    
    bary_coords, bary_weights = generate_simplex_grid(n, N_RES_SIMPLEX)
    
    if n > 1:
        verts = get_simplex_vertices(n)
        for p1, p2 in itertools.combinations(verts, 2):
            ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', alpha=0.3)
            
    if n > 0:
        verts = get_simplex_vertices(n)
        for i, idx in enumerate(selected_indices):
            v_pos = verts[i]
            label = vars_state[idx]['key']
            ax3d.text(v_pos[0], v_pos[1], v_pos[2], label, fontsize=9, weight='bold')

    plot_objs['sc_valid'] = ax3d.scatter([], [], [], s=20, cmap=plt.cm.jet, vmin=0, vmax=1)
    plot_objs['sc_nan'] = ax3d.scatter([], [], [], s=20, c='magenta')
    plot_objs['surf'] = None
    plot_objs['last_slice_state'] = {'pos': -1, 'axis': -1}

    ax2d.clear()
    if plot_objs['cbar']:
        try: plot_objs['cbar'].remove()
        except: pass
        plot_objs['cbar'] = None

    if n == 4:
        ax2d.set_title("2D Cross-section")
        u = np.linspace(0, 1, N_RES_GRID)
        v = np.linspace(0, 1, N_RES_GRID)
        U, V = np.meshgrid(u, v)
        plot_objs['grid_2d'] = (U, V)
        
        plot_objs['img'] = ax2d.imshow(np.zeros((N_RES_GRID, N_RES_GRID)), origin='lower', extent=[0,1,0,1], cmap=cmap_jet)
        plot_objs['cbar'] = fig.colorbar(plot_objs['img'], ax=ax2d, shrink=0.7)
    else:
        ax2d.text(0.5, 0.5, "Select exactly 4 vars", ha='center', va='center')
        ax2d.set_axis_off()
        plot_objs['img'] = None

def update_values(val=None):
    selected_indices = [i for i, v in enumerate(vars_state) if v['selected']]
    n = len(selected_indices)
    
    metric_mode = radio_metric.value_selected
    
    try: log_exp = float(text_log.text)
    except: log_exp = 2.0
    
    use_log = False
    if metric_mode == 'Equal Opportunity (Ratio)':
        check_log.labels[0].set_color('black')
        text_log.label.set_color('black')
        use_log = check_log.get_status()[0]
    else:
        check_log.labels[0].set_color('lightgray')
        text_log.label.set_color('lightgray')
        if check_log.get_status()[0]:
             check_log.set_active(0)
             return

    norm = None
    if metric_mode == 'Equal Opportunity (Ratio)':
        if use_log:
            norm = mcolors.LogNorm(vmin=10**(-log_exp), vmax=10**(log_exp))
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=2.0)
    elif metric_mode in ['Equalized Odds (Avg Diff)']:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    elif metric_mode == 'Global Recall':
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    alpha_val = 0.15 if check_transparency.get_status()[0] else 1.0

    # --- UPDATE 3D ---
    if n > 0:
        metric_inputs = {}
        w_idx = 0
        for i, var in enumerate(vars_state):
            key = var['key']
            if var['selected']:
                if w_idx < len(bary_weights):
                    metric_inputs[key] = bary_weights[w_idx]
                    w_idx += 1
            else:
                metric_inputs[key] = float(var['value'])

        vals = calculate_metric(metric_inputs, metric_mode)
        vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        mask_valid = ~np.isnan(vals)
        
        if np.any(mask_valid):
            p = bary_coords[mask_valid]
            v = vals[mask_valid]
            plot_objs['sc_valid']._offsets3d = (p[:,0], p[:,1], p[:,2])
            plot_objs['sc_valid'].set_array(v)
            plot_objs['sc_valid'].set_norm(norm)
            plot_objs['sc_valid'].set_alpha(alpha_val)
            plot_objs['sc_valid'].set_cmap(plt.cm.jet)
        else:
            plot_objs['sc_valid']._offsets3d = ([],[],[])

        if np.any(~mask_valid):
            p = bary_coords[~mask_valid]
            plot_objs['sc_nan']._offsets3d = (p[:,0], p[:,1], p[:,2])
            plot_objs['sc_nan'].set_alpha(alpha_val)
        else:
            plot_objs['sc_nan']._offsets3d = ([],[],[])
    else:
        plot_objs['sc_valid']._offsets3d = ([],[],[])
        plot_objs['sc_nan']._offsets3d = ([],[],[])

    # --- UPDATE 2D ---
    if n == 4 and plot_objs['img'] is not None:
        pos = slider_pos.val
        ax_sel = radio_axis.value_selected
        
        slice_state_changed = (pos != plot_objs['last_slice_state']['pos'] or 
                               ax_sel != plot_objs['last_slice_state']['axis'])
        
        U, V = plot_objs['grid_2d']
        if 'Z' in ax_sel:
            X_grid, Y_grid, Z_grid = U, V, np.full_like(U, pos)
            lbl_x, lbl_y = vars_state[selected_indices[1]]['key'], vars_state[selected_indices[2]]['key']
        elif 'Y' in ax_sel:
            X_grid, Z_grid, Y_grid = U, V, np.full_like(U, pos)
            lbl_x, lbl_y = vars_state[selected_indices[1]]['key'], vars_state[selected_indices[3]]['key']
        else:
            Y_grid, Z_grid, X_grid = U, V, np.full_like(U, pos)
            lbl_x, lbl_y = vars_state[selected_indices[2]]['key'], vars_state[selected_indices[3]]['key']

        w1, w2, w3, w4 = get_barycentric_for_slice(X_grid, Y_grid, Z_grid)
        eps = 1e-9
        mask_inside = (w1>=-eps) & (w2>=-eps) & (w3>=-eps) & (w4>=-eps)
        
        inputs_2d = {}
        for var in vars_state: inputs_2d[var['key']] = var['value']
        inputs_2d[vars_state[selected_indices[0]]['key']] = w1
        inputs_2d[vars_state[selected_indices[1]]['key']] = w2
        inputs_2d[vars_state[selected_indices[2]]['key']] = w3
        inputs_2d[vars_state[selected_indices[3]]['key']] = w4
        
        vals_2d = calculate_metric(inputs_2d, metric_mode)
        vals_2d = np.nan_to_num(vals_2d, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        v_masked = np.ma.masked_where(~mask_inside, vals_2d)
        
        plot_objs['img'].set_data(v_masked)
        plot_objs['img'].set_norm(norm)
        plot_objs['img'].set_cmap(cmap_jet)
        
        ax2d.set_xlabel(lbl_x)
        ax2d.set_ylabel(lbl_y)
        
        valid_v = vals_2d[mask_inside]
        if len(valid_v) > 0 and np.any(~np.isnan(valid_v)):
            m = np.nanmean(valid_v)
            s = np.nanstd(valid_v)
            stats_text.set_text(f"Mean: {m:.4f} | Std: {s:.4f}")
        else:
            stats_text.set_text("Mean: N/A")

        if slice_state_changed:
            if plot_objs['surf']: plot_objs['surf'].remove()
            X_s = np.where(mask_inside, X_grid, np.nan)
            Y_s = np.where(mask_inside, Y_grid, np.nan)
            Z_s = np.where(mask_inside, Z_grid, np.nan)
            plot_objs['surf'] = ax3d.plot_surface(X_s, Y_s, Z_s, color='red', alpha=0.3, rstride=10, cstride=10, shade=False)
            
            plot_objs['last_slice_state']['pos'] = pos
            plot_objs['last_slice_state']['axis'] = ax_sel

    fig.canvas.draw_idle()

def full_update(val=None):
    init_geometry()
    update_values()

# -------------------------------------------------
#  Event Handlers
# -------------------------------------------------
def make_callback(var_idx):
    def callback(label):
        current_sel = sum(v['selected'] for v in vars_state)
        is_now_checked = vars_state[var_idx]['checkbox'].get_status()[0]
        
        if is_now_checked and not vars_state[var_idx]['selected']:
            if current_sel >= 4:
                vars_state[var_idx]['checkbox'].set_active(0)
                print("Max 4 variables allowed.")
                return

        vars_state[var_idx]['selected'] = is_now_checked
        
        # Ukrywanie kontrolek
        if is_now_checked:
            vars_state[var_idx]['textbox'].ax.set_visible(False)
            vars_state[var_idx]['slider'].ax.set_visible(False)
        else:
            vars_state[var_idx]['textbox'].ax.set_visible(True)
            vars_state[var_idx]['slider'].ax.set_visible(True)
            
        full_update()
    return callback

for i, var in enumerate(vars_state):
    var['checkbox'].on_clicked(make_callback(i))
    
    def make_s_cb(i):
        def cb(val):
            vars_state[i]['value'] = val
            vars_state[i]['textbox'].set_val(f"{val:.2f}")
            update_values()
        return cb
    
    def make_t_cb(i):
        def cb(text):
            try: v = float(text)
            except: return
            if v < 0: v = 0
            if v > 1: v = 1
            vars_state[i]['value'] = v
            vars_state[i]['slider'].set_val(v)
        return cb

    var['slider'].on_changed(make_s_cb(i))
    var['textbox'].on_submit(make_t_cb(i))

radio_metric.on_clicked(update_values)
slider_pos.on_changed(update_values)
check_transparency.on_clicked(update_values)
check_log.on_clicked(update_values)
text_log.on_submit(update_values)
radio_axis.on_clicked(update_values)

def upd_res(txt):
    global N_RES_SIMPLEX
    try: N_RES_SIMPLEX = int(txt)
    except: return
    full_update()
text_res.on_submit(upd_res)

# Start
full_update()
plt.show()