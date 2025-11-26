import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# -------------------------------------------------
#  Definicje wierzchołków
# -------------------------------------------------
TP_p = np.array([0.0, 0.0, 0.0])
FN_p = np.array([1.0, 0.0, 1.0])
TP_u = np.array([1.0, 1.0, 0.0])
FN_u = np.array([0.0, 1.0, 1.0])

vertices = [TP_p, FN_p, TP_u, FN_u]
labels = ["TP_p", "FN_p", "TP_u", "FN_u"]

# -------------------------------------------------
#  MAPY KOLORÓW
# -------------------------------------------------
base_cmap = plt.cm.jet

# 1. Mapa dla 2D (imshow)
cmap_2d = base_cmap.copy()
cmap_2d.set_bad(color='white')

# 2. Mapa dla 3D (scatter)
cmap_3d = base_cmap.copy()

# -------------------------------------------------
#  Funkcje obliczeniowe
# -------------------------------------------------
def calculate_metric(tpp, fnp, tpu, fnu, mode):
    p_prot = tpp + fnp
    p_unprot = tpu + fnu
    
    tpr_prot = np.divide(tpp, p_prot, out=np.full_like(tpp, np.nan), where=p_prot!=0)
    tpr_unprot = np.divide(tpu, p_unprot, out=np.full_like(tpu, np.nan), where=p_unprot!=0)

    if mode == 'Eq. Opp. (Diff)':
        return tpr_prot - tpr_unprot

    elif mode == 'Eq. Opp. (Ratio)':
        # Ratio: TPR_p / TPR_u
        return np.divide(tpr_prot, tpr_unprot, out=np.full_like(tpr_prot, np.nan), where=tpr_unprot!=0)

    elif mode == 'Global Recall':
        tp_total = tpp + tpu
        p_total = p_prot + p_unprot
        return np.divide(tp_total, p_total, out=np.full_like(tp_total, np.nan), where=p_total!=0)

    else:
        return np.zeros_like(tpp)

def f_point(tpp, fnp, tpu, fnu):
    w = np.array([tpp, fnp, tpu, fnu])
    if w.sum() == 0: return np.zeros(3)
    w = w / w.sum()
    return w[0]*TP_p + w[1]*FN_p + w[2]*TP_u + w[3]*FN_u

def get_barycentric(x, y, z):
    w_sum = (x + y + z) / 2.0
    w_tpp = 1.0 - w_sum
    w_fnp = w_sum - y
    w_tpu = w_sum - z
    w_fnu = w_sum - x
    return w_tpp, w_fnp, w_tpu, w_fnu

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
ax3d.set_title("Widok 3D (Przestrzeń Pozytywów)", fontsize=12)
ax3d.set_axis_off()

for idx1, idx2 in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
    P, Q = vertices[idx1], vertices[idx2]
    ax3d.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], color='black', alpha=0.4, linewidth=1)

sc_valid = ax3d.scatter([], [], [], s=15, cmap=cmap_3d)
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

# Inicjalizacja imshow
img = ax2d.imshow(np.ma.zeros((N_2d, N_2d)), origin='lower', extent=[0,1,0,1], cmap=cmap_2d)
cbar = fig.colorbar(img, ax=ax2d, shrink=0.7)

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
    
    # Logika Checkboxa Log Scale i Pola Tekstowego
    log_label = check_log.labels[0]
    
    # Pobieramy potęgę z pola tekstowego (np. 1 -> 10^-1 do 10^1)
    try:
        log_exp = float(text_log.text)
    except ValueError:
        log_exp = 1.0 # Domyślnie
    
    if metric_mode == 'Eq. Opp. (Ratio)':
        # Aktywacja
        log_label.set_color('black')
        text_log.label.set_color('black')
        text_log.set_active(True) # Odblokowanie pola tekstowego
        
        use_log = check_log.get_status()[0]
    else:
        # Dezaktywacja
        log_label.set_color('lightgray')
        text_log.label.set_color('lightgray')
        # text_log.set_active(False) # Można zablokować, ale matplotlib < 3.3 nie ma set_active dla TextBox wprost, zostawiamy aktywne wizualnie wyszarzone
        
        # Wymuszenie wyłączenia log scale
        if check_log.get_status()[0]:
            check_log.set_active(0) 
            return
        use_log = False

    # Checkbox "Transparent"
    is_transparent = check_transparency.get_status()[0]
    alpha_val = 0.15 if is_transparent else 1.0

    # 2. Ustalanie Normalizacji
    norm = None
    
    if metric_mode == 'Eq. Opp. (Ratio)':
        if use_log:
            # Skala Logarytmiczna: 10^-x do 10^x
            vmin_log = 10**(-log_exp)
            vmax_log = 10**(log_exp)
            norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log)
        else:
            # Skala Liniowa
            norm = mcolors.Normalize(vmin=0.0, vmax=2.0)
            
    elif metric_mode == 'Eq. Opp. (Diff)':
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    else: # Global Recall
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 3. Update 3D Scatter
    tpp, fnp, tpu, fnu = bary_weights_array
    vals = calculate_metric(tpp, fnp, tpu, fnu, metric_mode)
    
    vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    mask_nan = np.isnan(vals)
    mask_valid = ~mask_nan
    
    pts_valid = pts_3d_coords[mask_valid]
    vals_valid = vals[mask_valid]
    pts_nan = pts_3d_coords[mask_nan]
    
    sc_valid.remove()
    sc_nan.remove()
    
    if len(pts_valid) > 0:
        sc_valid = ax3d.scatter(pts_valid[:,0], pts_valid[:,1], pts_valid[:,2], 
                                s=15, c=vals_valid, cmap=cmap_3d, norm=norm, alpha=alpha_val)
    else:
        sc_valid = ax3d.scatter([], [], [], s=15, cmap=cmap_3d)
        
    if len(pts_nan) > 0:
        sc_nan = ax3d.scatter(pts_nan[:,0], pts_nan[:,1], pts_nan[:,2], 
                              s=15, c='#FF00FF', alpha=alpha_val)
    else:
        sc_nan = ax3d.scatter([], [], [], s=15, c='#FF00FF')

    # 4. Update 2D Slice
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
        
    tpp, fnp, tpu, fnu = get_barycentric(X_grid, Y_grid, Z_grid)
    eps = 1e-9
    mask_inside = (tpp >= -eps) & (fnp >= -eps) & (tpu >= -eps) & (fnu >= -eps)
    
    values = calculate_metric(tpp, fnp, tpu, fnu, metric_mode)
    values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # Statystyki
    valid_slice_values = values[mask_inside]
    if len(valid_slice_values) > 0 and np.any(~np.isnan(valid_slice_values)):
        mean_val = np.nanmean(valid_slice_values)
        std_val = np.nanstd(valid_slice_values)
        stats_str = f"Mean: {mean_val:.4f}  |  Std Dev: {std_val:.4f}"
    else:
        stats_str = "Mean: N/A  |  Std Dev: N/A"
    stats_text.set_text(stats_str)
    
    values_masked = np.ma.masked_where(~mask_inside, values)
    img.set_data(values_masked)
    img.set_norm(norm) 
    
    ax2d.set_xlabel(xlabel, fontsize=10)
    ax2d.set_ylabel(ylabel, fontsize=10)
    
    cbar.update_normal(img)
    
    # 5. Update 3D Plane
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

# Oś Cięcia
ax_radio_axis = plt.axes([0.05, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_axis = RadioButtons(ax_radio_axis, ['Z (TPp-TPu)', 'Y (TPp-FNp)', 'X (TPp-FNu)'], active=0)
ax_radio_axis.set_title("Oś Cięcia", fontsize=9)

# Metryka
ax_radio_metric = plt.axes([0.18, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_metric = RadioButtons(ax_radio_metric, ['Eq. Opp. (Diff)', 'Eq. Opp. (Ratio)', 'Global Recall'], active=0)
ax_radio_metric.set_title("Metryka", fontsize=9)

# Kontrolki (Checkboxy)
ax_controls = plt.axes([0.31, 0.02, 0.16, 0.18], facecolor=bg_color)
ax_controls.axis('off')

# Checkbox: Transparent
ax_check_trans = plt.axes([0.31, 0.13, 0.12, 0.05], frameon=False)
check_transparency = CheckButtons(ax_check_trans, ['Transparent'], [True])

# Checkbox: Log Scale (NOWY)
ax_check_log = plt.axes([0.31, 0.08, 0.12, 0.05], frameon=False)
check_log = CheckButtons(ax_check_log, ['Log Scale'], [False])

# TextBox: Log Range (Power of 10) - Umieszczony obok checkboxa
ax_text_log = plt.axes([0.43, 0.085, 0.04, 0.04])
text_log = TextBox(ax_text_log, '', initial="2") # Pusty label, bo jest oczywiste obok log scale
text_log.label.set_size(9)

# TextBox: Resolution
ax_box = plt.axes([0.32, 0.03, 0.05, 0.04])
text_box = TextBox(ax_box, 'Res: ', initial=str(current_N))
text_box.label.set_size(9)

# Slidery
ax_slider_pos = plt.axes([0.50, 0.12, 0.43, 0.03])
slider_pos = Slider(ax_slider_pos, 'Poz. Cięcia ', 0.0, 1.0, valinit=0.5, color='red')
slider_pos.label.set_size(9)

ax_slider_roll = plt.axes([0.50, 0.07, 0.43, 0.03])
slider_roll = Slider(ax_slider_roll, 'Roll (3D) ', -180, 180, valinit=0, color='orange')
slider_roll.label.set_size(9)

radio_axis.on_clicked(update)
radio_metric.on_clicked(update)
slider_pos.on_changed(update)
slider_roll.on_changed(update_roll)
check_transparency.on_clicked(update)
check_log.on_clicked(update)
text_box.on_submit(update_resolution)
text_log.on_submit(update) # Update po wpisaniu potęgi

update()
plt.show()