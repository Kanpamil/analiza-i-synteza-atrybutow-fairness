import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons
import matplotlib.cm as cm

# -------------------------------------------------
#  Definicje wierzchołków
# -------------------------------------------------
TP = np.array([0.0, 0.0, 0.0])
FP = np.array([1.0, 0.0, 1.0])
TN = np.array([1.0, 1.0, 0.0])
FN = np.array([0.0, 1.0, 1.0])
vertices = [TP, FP, TN, FN]
labels = ["TP", "FP", "TN", "FN"]

# -------------------------------------------------
#  MAPY KOLORÓW
# -------------------------------------------------
base_cmap = plt.cm.jet

# 1. Mapa dla 2D (imshow) - NaN ma być biały (tło)
cmap_2d = base_cmap.copy()
cmap_2d.set_bad(color='white')

# 2. Mapa dla 3D (scatter) - valid points
cmap_3d = base_cmap.copy()

# -------------------------------------------------
#  Funkcje obliczeniowe
# -------------------------------------------------
def calculate_metric(tp, fp, tn, fn, mode):
    if mode == 'Accuracy':
        num = tp + tn
        den = tp + fp + tn + fn
    elif mode == 'Precision':
        num = tp
        den = tp + fp
    elif mode == 'Recall':
        num = tp
        den = tp + fn
    elif mode == 'F1 Score':
        num = 2 * tp
        den = 2 * tp + fp + fn
    else:
        return np.zeros_like(tp)

    # Zwraca NaN tam gdzie dzielenie przez zero
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den!=0)

def f_point(tp, fp, tn, fn):
    w = np.array([tp, fp, tn, fn])
    if w.sum() == 0: return np.zeros(3)
    w = w / w.sum()
    return w[0]*TP + w[1]*FP + w[2]*TN + w[3]*FN

def get_barycentric(x, y, z):
    w_sum = (x + y + z) / 2.0
    w_tp = 1.0 - w_sum
    w_fp = w_sum - y
    w_tn = w_sum - z
    w_fn = w_sum - x
    return w_tp, w_fp, w_tn, w_fn

def generate_point_cloud(N):
    coords = []
    weights = []
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                l = N - i - j - k
                w = np.array([i,j,k,l]) / N
                coords.append(f_point(*w))
                weights.append(w)
    return np.array(coords), np.array(weights).T

# -------------------------------------------------
#  Inicjalizacja Stanu
# -------------------------------------------------
current_N = 12
pts_3d_coords, bary_weights_array = generate_point_cloud(current_N)

# -------------------------------------------------
#  Konfiguracja Okna
# -------------------------------------------------
fig = plt.figure(figsize=(14, 9))
plt.subplots_adjust(bottom=0.25, wspace=0.25, left=0.05, right=0.95)

# --- PANEL 3D ---
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_title("Widok 3D", fontsize=12)
ax3d.set_axis_off()

for idx1, idx2 in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
    P, Q = vertices[idx1], vertices[idx2]
    ax3d.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], color='black', alpha=0.4, linewidth=1)

# Inicjalizacja pustych scatterów, wypełnimy w update()
sc_valid = ax3d.scatter([], [], [], s=15, cmap=cmap_3d, vmin=0, vmax=1)
sc_nan = ax3d.scatter([], [], [], s=15, c='magenta') 

for v, label in zip(vertices, labels):
    ax3d.text(v[0], v[1], v[2]+0.05, label, fontsize=11, weight='bold')

# --- PANEL 2D ---
ax2d = fig.add_subplot(122)
ax2d.set_title("Przekrój 2D", fontsize=12)

N_2d = 150
u = np.linspace(0, 1, N_2d)
v = np.linspace(0, 1, N_2d)
U, V = np.meshgrid(u, v)

# Imshow 2D
img = ax2d.imshow(np.ma.zeros((N_2d, N_2d)), origin='lower', extent=[0,1,0,1], vmin=0, vmax=1, cmap=cmap_2d)
cbar = fig.colorbar(img, ax=ax2d, shrink=0.7)

# --- NOWE: Obiekt tekstowy dla statystyk ---
# Umieszczamy go pod wykresem 2D (używając transform=ax2d.transAxes)
stats_text = ax2d.text(0.5, -0.15, "", transform=ax2d.transAxes, ha='center', va='top', fontsize=11, fontweight='bold', color='#333333')

surf_plot = None

# -------------------------------------------------
#  Logika Aktualizacji
# -------------------------------------------------
def update(val=None):
    global surf_plot, sc_valid, sc_nan
    
    # 1. Widget values
    axis_mode = radio_axis.value_selected
    metric_mode = radio_metric.value_selected
    pos = slider_pos.val
    is_transparent = check_transparency.get_status()[0]
    alpha_valid = 0.15 if is_transparent else 1.0
    alpha_nan = 1.0 
    
    # 2. Update 3D Scatter (Split into Valid and NaN)
    wtp, wfp, wtn, wfn = bary_weights_array
    vals = calculate_metric(wtp, wfp, wtn, wfn, metric_mode)
    
    mask_nan = np.isnan(vals)
    mask_valid = ~mask_nan
    
    pts_valid = pts_3d_coords[mask_valid]
    vals_valid = vals[mask_valid]
    pts_nan = pts_3d_coords[mask_nan]
    
    sc_valid.remove()
    sc_nan.remove()
    
    if len(pts_valid) > 0:
        sc_valid = ax3d.scatter(pts_valid[:,0], pts_valid[:,1], pts_valid[:,2], 
                                s=15, c=vals_valid, cmap=cmap_3d, alpha=alpha_valid, vmin=0, vmax=1)
    else:
        sc_valid = ax3d.scatter([], [], [], s=15, cmap=cmap_3d, vmin=0, vmax=1)
        
    if len(pts_nan) > 0:
        sc_nan = ax3d.scatter(pts_nan[:,0], pts_nan[:,1], pts_nan[:,2], 
                              s=15, c='#FF00FF', alpha=alpha_nan)
    else:
        sc_nan = ax3d.scatter([], [], [], s=15, c='#FF00FF')

    # 3. Update 2D Slice
    if 'Z' in axis_mode: 
        X_grid, Y_grid = U, V
        Z_grid = np.full_like(U, pos)
        xlabel, ylabel = 'X', 'Y'
    elif 'Y' in axis_mode: 
        X_grid, Z_grid = U, V
        Y_grid = np.full_like(U, pos)
        xlabel, ylabel = 'X', 'Z'
    elif 'X' in axis_mode: 
        Y_grid, Z_grid = U, V
        X_grid = np.full_like(U, pos)
        xlabel, ylabel = 'Y', 'Z'
        
    wtp, wfp, wtn, wfn = get_barycentric(X_grid, Y_grid, Z_grid)
    eps = 1e-9
    mask_inside = (wtp >= -eps) & (wfp >= -eps) & (wtn >= -eps) & (wfn >= -eps)
    
    values = calculate_metric(wtp, wfp, wtn, wfn, metric_mode)
    
    # --- NOWE: Obliczanie statystyk ---
    # Bierzemy pod uwagę tylko punkty wewnątrz czworościanu (mask_inside)
    valid_slice_values = values[mask_inside]
    
    # Może się zdarzyć, że slice jest pusty lub zawiera same NaNs
    if len(valid_slice_values) > 0:
        # np.nanmean ignoruje NaN, co jest pożądane (bo NaN to błąd 0/0)
        mean_val = np.nanmean(valid_slice_values)
        std_val = np.nanstd(valid_slice_values)
        stats_str = f"Mean: {mean_val:.4f}  |  Std Dev: {std_val:.4f}"
    else:
        stats_str = "Mean: N/A  |  Std Dev: N/A"

    stats_text.set_text(stats_str)
    
    # Update obrazu
    values_masked = np.ma.masked_where(~mask_inside, values)
    img.set_data(values_masked)
    ax2d.set_xlabel(xlabel, fontsize=10)
    ax2d.set_ylabel(ylabel, fontsize=10)
    cbar.set_label(metric_mode, fontsize=10)
    
    # 4. Update 3D Plane
    if surf_plot is not None:
        surf_plot.remove()
        
    X_surf = np.where(mask_inside, X_grid, np.nan)
    Y_surf = np.where(mask_inside, Y_grid, np.nan)
    Z_surf = np.where(mask_inside, Z_grid, np.nan)
    
    surf_plot = ax3d.plot_surface(X_surf, Y_surf, Z_surf, color='red', alpha=0.6, 
                                  rstride=10, cstride=10, shade=False)
    
    fig.canvas.draw_idle()

def update_resolution(text):
    global pts_3d_coords, bary_weights_array
    try:
        new_N = int(text)
        if new_N < 1 or new_N > 60: return
    except ValueError:
        return

    pts_3d_coords, bary_weights_array = generate_point_cloud(new_N)
    update() 

def update_roll(val):
    try:
        ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim, roll=slider_roll.val)
    except TypeError:
        ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim)
    fig.canvas.draw_idle()

# -------------------------------------------------
#  Widgety
# -------------------------------------------------
bg_color = '#f0f0f0'

ax_radio_axis = plt.axes([0.05, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_axis = RadioButtons(ax_radio_axis, ['Z (TP-TN)', 'Y (TP-FP)', 'X (TP-FN)'], active=0)
ax_radio_axis.set_title("Oś Cięcia", fontsize=9)

ax_radio_metric = plt.axes([0.18, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_metric = RadioButtons(ax_radio_metric, ['Accuracy', 'Precision', 'Recall', 'F1 Score'], active=0)
ax_radio_metric.set_title("Metryka", fontsize=9)

ax_controls = plt.axes([0.32, 0.02, 0.10, 0.18], facecolor=bg_color)
ax_controls.axis('off')

ax_check = plt.axes([0.32, 0.12, 0.10, 0.08], frameon=False)
check_transparency = CheckButtons(ax_check, ['Transparent'], [True])

ax_box = plt.axes([0.33, 0.05, 0.05, 0.04])
text_box = TextBox(ax_box, 'Res: ', initial=str(current_N))
text_box.label.set_size(9)

ax_slider_pos = plt.axes([0.48, 0.12, 0.45, 0.03])
slider_pos = Slider(ax_slider_pos, 'Poz. Cięcia ', 0.0, 1.0, valinit=0.5, color='red')
slider_pos.label.set_size(9)

ax_slider_roll = plt.axes([0.48, 0.07, 0.45, 0.03])
slider_roll = Slider(ax_slider_roll, 'Roll (3D) ', -180, 180, valinit=0, color='orange')
slider_roll.label.set_size(9)

# Callbacki
radio_axis.on_clicked(update)
radio_metric.on_clicked(update)
slider_pos.on_changed(update)
slider_roll.on_changed(update_roll)
check_transparency.on_clicked(update)
text_box.on_submit(update_resolution)

update()
plt.show()