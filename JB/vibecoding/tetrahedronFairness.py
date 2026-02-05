import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons, Button
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime

# -------------------------------------------------
#   Definicje wierzchołków
# -------------------------------------------------
TP_p = np.array([0.0, 0.0, 0.0])
FN_p = np.array([1.0, 0.0, 1.0])
TP_u = np.array([1.0, 1.0, 0.0])
FN_u = np.array([0.0, 1.0, 1.0])

vertices = [TP_p, FN_p, TP_u, FN_u]
labels = ["TP_p", "FN_p", "TP_u", "FN_u"]

# -------------------------------------------------
#   MAPY KOLORÓW
# -------------------------------------------------
cmap_full_jet = plt.cm.jet.copy()
cmap_full_jet.set_bad(color='white')

jet_sample = plt.cm.jet(np.linspace(0.5, 1.0, 256))
cmap_green_red = mcolors.LinearSegmentedColormap.from_list("jet_green_red", jet_sample)
cmap_green_red.set_bad(color='white')

cmap_ratio_custom = plt.cm.viridis.copy()
cmap_ratio_custom.set_bad(color='white')

# -------------------------------------------------
#   Funkcje obliczeniowe
# -------------------------------------------------
def calculate_metric(tpp, fnp, tpu, fnu, mode, use_abs=False):
    p_prot = tpp + fnp
    p_unprot = tpu + fnu
    
    val = np.zeros_like(tpp)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr_prot = np.divide(tpp, p_prot, out=np.full_like(tpp, np.nan), where=p_prot!=0)
        tpr_unprot = np.divide(tpu, p_unprot, out=np.full_like(tpu, np.nan), where=p_unprot!=0)

        if mode == 'Eq. Opp. (Diff)':
            val = tpr_prot - tpr_unprot
        elif mode == 'Eq. Opp. (Ratio)':
            val = np.divide(tpr_prot, tpr_unprot, out=np.full_like(tpr_prot, np.nan), where=tpr_unprot!=0)
        elif mode == 'Global Recall':
            tp_total = tpp + tpu
            p_total = p_prot + p_unprot
            val = np.divide(tp_total, p_total, out=np.full_like(tp_total, np.nan), where=p_total!=0)
        elif mode == 'Eq. Opp. (Norm)':
            denom = tpr_prot + tpr_unprot
            val = np.divide(tpr_prot, denom, out=np.full_like(tpr_prot, np.nan), where=denom!=0)
            
    if use_abs:
        val = np.abs(val)
        
    return val

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

# --- GENERATOR PUNKTÓW ---
def generate_point_cloud(N, surface_only=False):
    coords = []
    weights = []
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                l = N - i - j - k
                
                if surface_only:
                    if not (i == 0 or j == 0 or k == 0 or l == 0):
                        continue

                w = np.array([i,j,k,l]) / N
                coords.append(f_point(*w))
                weights.append(w)
    return np.array(coords), np.array(weights).T

# -------------------------------------------------
#   Inicjalizacja Stanu
# -------------------------------------------------
current_N = 12
pts_3d_coords, bary_weights_array = generate_point_cloud(current_N, surface_only=False)

surf_plot = None
contours = None
contour_labels = []
cbar_2d = None
sc_main = None 

# -------------------------------------------------
#   Konfiguracja Okna Głównego
# -------------------------------------------------
fig = plt.figure(figsize=(15, 9))
plt.subplots_adjust(bottom=0.25, wspace=0.3, left=0.05, right=0.95)

current_norm = None
current_levels = None
current_cmap = None
current_vals_valid = None
current_pts_valid = None
current_pts_nan = None
current_is_transparent = False

# --- WYKRES 3D ---
ax3d = fig.add_subplot(121, projection='3d')
ax3d.view_init(elev=-68, azim=-142, roll=56)
ax3d.set_title("Widok 3D (Przestrzeń Pozytywów)", fontsize=12)
ax3d.set_axis_off()

ax_cbar_3d = fig.add_axes([0.48, 0.35, 0.015, 0.4]) 

def draw_tetrahedron_edges(ax):
    for idx1, idx2 in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
        P, Q = vertices[idx1], vertices[idx2]
        ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], color='black', alpha=0.4, linewidth=1)
    
    centroid = np.mean(vertices, axis=0)
    offset_factor = 0.25
    for v, label in zip(vertices, labels):
        direction = v - centroid
        pos = v + direction * offset_factor
        ax.text(pos[0], pos[1], pos[2], label, fontsize=11, weight='bold', ha='center', va='center')

draw_tetrahedron_edges(ax3d)
sc_main = ax3d.scatter([], [], [], s=15)

# --- WYKRES 2D ---
ax2d = fig.add_subplot(122)
ax2d.set_title("Przekrój 2D", fontsize=12)

N_2d = 150
u = np.linspace(0, 1, N_2d)
v = np.linspace(0, 1, N_2d)
U, V = np.meshgrid(u, v)

img = ax2d.imshow(np.ma.zeros((N_2d, N_2d)), origin='lower', extent=[0,1,0,1], cmap=cmap_full_jet)
cbar_2d = fig.colorbar(img, ax=ax2d, shrink=0.7)

stats_text = ax2d.text(0.5, -0.15, "", transform=ax2d.transAxes, ha='center', va='top', fontsize=11, fontweight='bold', color='#333333')

# -------------------------------------------------
#   Logika Parametrów
# -------------------------------------------------
def get_visualization_params():
    metric_mode = radio_metric.value_selected
    is_abs = check_abs.get_status()[0]
    
    try:
        log_exp = float(text_log.text)
    except ValueError:
        log_exp = 2.0 
    
    use_log = False
    if metric_mode == 'Eq. Opp. (Ratio)':
        use_log = check_log.get_status()[0]

    norm = None
    levels = None
    cmap = cmap_full_jet 
    
    if metric_mode == 'Eq. Opp. (Ratio)':
        cmap = cmap_ratio_custom
        if use_log:
            vmin_log = 10**(-log_exp)
            vmax_log = 10**(log_exp)
            norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log)
            levels = np.logspace(-log_exp, log_exp, 15)
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=2.0)
            levels = np.linspace(0.1, 1.9, 10)
            
    elif metric_mode == 'Eq. Opp. (Diff)':
        if is_abs:
            cmap = cmap_green_red
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            levels = np.linspace(0.05, 0.95, 10)
        else:
            cmap = cmap_full_jet
            norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
            levels = np.linspace(-0.9, 0.9, 10)
        
    elif metric_mode == 'Eq. Opp. (Norm)':
        cmap = cmap_green_red
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        levels = np.linspace(0.1, 0.9, 9)
        
    else: # Global Recall
        cmap = cmap_green_red
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        levels = np.linspace(0.1, 0.9, 9)
        
    return metric_mode, norm, levels, cmap, log_exp, is_abs

# -------------------------------------------------
#   Główna Funkcja Update
# -------------------------------------------------
def update(val=None):
    global surf_plot, sc_main, contours, contour_labels, cbar_2d
    global current_norm, current_levels, current_vals_valid, current_pts_valid, current_pts_nan, current_is_transparent, current_cmap
    
    axis_mode = radio_axis.value_selected
    pos = slider_pos.val
    
    metric_mode = radio_metric.value_selected
    log_label = check_log.labels[0]
    
    # Logika UI dla Log Scale
    if metric_mode == 'Eq. Opp. (Ratio)':
        log_label.set_color('black')
        text_log.label.set_color('black')
        text_log.set_active(True)
    else:
        log_label.set_color('lightgray')
        text_log.label.set_color('lightgray')
        if check_log.get_status()[0]:
            check_log.set_active(0) 

    metric_mode, norm, levels, cmap, _, is_abs = get_visualization_params()
    
    current_norm = norm
    current_levels = levels
    current_cmap = cmap
    current_is_transparent = check_transparency.get_status()[0]
    alpha_val = 0.15 if current_is_transparent else 1.0
    
    show_plane = check_plane.get_status()[0]
    
    # 1. Update 3D Points
    tpp, fnp, tpu, fnu = bary_weights_array
    vals = calculate_metric(tpp, fnp, tpu, fnu, metric_mode, use_abs=is_abs)
    vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    mask_nan = np.isnan(vals)
    mask_valid = ~mask_nan
    
    pts_valid = pts_3d_coords[mask_valid]
    vals_valid = vals[mask_valid]
    pts_nan = pts_3d_coords[mask_nan]
    
    current_pts_valid = pts_valid
    current_vals_valid = vals_valid
    current_pts_nan = pts_nan
    
    # --- Łączenie punktów ---
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    if len(pts_valid) > 0:
        rgba_valid = mapper.to_rgba(vals_valid)
        rgba_valid[:, 3] = alpha_val
    else:
        rgba_valid = np.zeros((0, 4))

    if len(pts_nan) > 0:
        rgba_nan = np.tile(mcolors.to_rgba('#FF00FF'), (len(pts_nan), 1))
        rgba_nan[:, 3] = alpha_val
    else:
        rgba_nan = np.zeros((0, 4))
    
    if len(pts_valid) > 0 and len(pts_nan) > 0:
        all_pts = np.vstack((pts_valid, pts_nan))
        all_colors = np.vstack((rgba_valid, rgba_nan))
    elif len(pts_valid) > 0:
        all_pts = pts_valid
        all_colors = rgba_valid
    elif len(pts_nan) > 0:
        all_pts = pts_nan
        all_colors = rgba_nan
    else:
        all_pts = np.zeros((0, 3))
        all_colors = np.zeros((0, 4))

    if sc_main:
        sc_main.remove()
        
    if len(all_pts) > 0:
        sc_main = ax3d.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], 
                               s=15, c=all_colors, depthshade=True)
    else:
        sc_main = ax3d.scatter([], [], [], s=15)

    ax_cbar_3d.clear() 
    sm_3d = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_3d.set_array([]) 
    fig.colorbar(sm_3d, cax=ax_cbar_3d)

    # 2. Update 2D Slice
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
    
    values = calculate_metric(tpp, fnp, tpu, fnu, metric_mode, use_abs=is_abs)
    values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
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
    img.set_cmap(cmap)
    img.set_norm(norm) 
    
    if cbar_2d:
        cbar_2d.remove()
    cbar_2d = fig.colorbar(img, ax=ax2d, shrink=0.7)
    
    if contours is not None:
        for c in contours.collections: c.remove()
    for txt in contour_labels: txt.remove()
    contour_labels.clear()

    if np.any(~values_masked.mask):
        try:
            contours = ax2d.contour(U, V, values_masked, levels=levels, 
                                    colors='white', linewidths=1.5, alpha=0.9)
            clbls = ax2d.clabel(contours, inline=True, fontsize=9, fmt='%1.1f', colors='white')
            contour_labels.extend(clbls)
        except:
            contours = None

    ax2d.set_xlabel(xlabel, fontsize=10)
    ax2d.set_ylabel(ylabel, fontsize=10)
    
    # 3. Update Plane
    if surf_plot is not None: 
        try: surf_plot.remove()
        except: pass 
        surf_plot = None 

    if show_plane:
        X_surf = np.where(mask_inside, X_grid, np.nan)
        Y_surf = np.where(mask_inside, Y_grid, np.nan)
        Z_surf = np.where(mask_inside, Z_grid, np.nan)
        
        surf_plot = ax3d.plot_surface(X_surf, Y_surf, Z_surf, color='red', alpha=1.0, 
                                      rstride=10, cstride=10, shade=False)
    
    fig.canvas.draw_idle()

def update_point_cloud_data(val=None):
    global pts_3d_coords, bary_weights_array
    try:
        new_N = int(text_box.text)
        if new_N < 1 or new_N > 60: return
    except ValueError:
        return
    
    is_surface_only = check_surface.get_status()[0]
    
    pts_3d_coords, bary_weights_array = generate_point_cloud(new_N, surface_only=is_surface_only)
    update()

def update_roll(val):
    try:
        ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim, roll=slider_roll.val)
    except TypeError:
        ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim)
    fig.canvas.draw_idle()

# -------------------------------------------------
#   FUNKCJE EKSPORTU (ZOPTYMALIZOWANE ROZMIAROWO)
# -------------------------------------------------
def generate_filename(prefix, extension):
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def export_3d(file_format):
    print(f"Rozpoczynam zapis wykresu 3D ({file_format})...")
    
    # 1. Niższe DPI - 150 wystarczy do podglądu, a drastycznie zmniejsza plik
    dpi_val = 150
    
    # 2. Mniejszy rozmiar fizyczny figury (mniej pikseli do wyrenderowania)
    export_fig = plt.figure(figsize=(6, 5)) 
    
    export_ax = export_fig.add_axes([0.0, 0.0, 0.90, 1.0], projection='3d')
    export_ax.set_axis_off()
    
    try:
        current_dist = ax3d.dist
    except AttributeError:
        current_dist = 10 

    current_elev = ax3d.elev
    current_azim = ax3d.azim
    current_roll = slider_roll.val 

    export_ax.view_init(elev=current_elev, azim=current_azim, roll=current_roll)
    try:
        export_ax.dist = current_dist 
    except AttributeError:
        pass 

    # Krawędzie wektorowe (będą ostre)
    draw_tetrahedron_edges(export_ax)
    
    metric_mode, norm, _, cmap, _, _ = get_visualization_params()
    alpha_val = 0.15 if current_is_transparent else 1.0
    
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    pts_to_export = []
    colors_to_export = []
    
    if len(current_pts_valid) > 0:
        rgba_valid = mapper.to_rgba(current_vals_valid)
        rgba_valid[:, 3] = alpha_val
        pts_to_export.append(current_pts_valid)
        colors_to_export.append(rgba_valid)
        
    if len(current_pts_nan) > 0:
        rgba_nan = np.tile(mcolors.to_rgba('#FF00FF'), (len(current_pts_nan), 1))
        rgba_nan[:, 3] = alpha_val
        pts_to_export.append(current_pts_nan)
        colors_to_export.append(rgba_nan)
        
    if len(pts_to_export) > 0:
        all_pts_exp = np.vstack(pts_to_export)
        all_colors_exp = np.vstack(colors_to_export)
        
        # 3. rasterized=True - to klucz do małych PDFów.
        # Punkty są zamieniane na bitmapę wewnątrz PDF, a nie trzymane jako wektory.
        export_ax.scatter(all_pts_exp[:,0], all_pts_exp[:,1], all_pts_exp[:,2], 
                          s=20, c=all_colors_exp, depthshade=True,
                          rasterized=True) 

    show_plane = check_plane.get_status()[0]
    if show_plane:
        axis_mode = radio_axis.value_selected
        pos = slider_pos.val
        if 'Z' in axis_mode: 
            X_grid, Y_grid = U, V
            Z_grid = np.full_like(U, pos)
        elif 'Y' in axis_mode: 
            X_grid, Z_grid = U, V
            Y_grid = np.full_like(U, pos)
        elif 'X' in axis_mode: 
            Y_grid, Z_grid = U, V
            X_grid = np.full_like(U, pos)
        tpp, fnp, tpu, fnu = get_barycentric(X_grid, Y_grid, Z_grid)
        eps = 1e-9
        mask_inside = (tpp >= -eps) & (fnp >= -eps) & (tpu >= -eps) & (fnu >= -eps)
        X_surf = np.where(mask_inside, X_grid, np.nan)
        Y_surf = np.where(mask_inside, Y_grid, np.nan)
        Z_surf = np.where(mask_inside, Z_grid, np.nan)
        
        # Płaszczyzna też rasteryzowana
        export_ax.plot_surface(X_surf, Y_surf, Z_surf, color='red', alpha=1.0, 
                               rstride=5, cstride=5, shade=False,
                               rasterized=True)

    if len(current_pts_valid) > 0:
        cbar_ax_exp = export_fig.add_axes([0.90, 0.2, 0.02, 0.6]) 
        sm_exp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_exp.set_array([])
        export_fig.colorbar(sm_exp, cax=cbar_ax_exp)

    fname = generate_filename("Export_3D", file_format)
    export_fig.savefig(fname, bbox_inches='tight', pad_inches=0, format=file_format, dpi=dpi_val)
    plt.close(export_fig)
    print(f"Zapisano (Lekki plik): {fname}")

def export_2d(file_format):
    print(f"Rozpoczynam zapis wykresu 2D ({file_format})...")
    
    dpi_val = 150 
    
    export_fig = plt.figure(figsize=(6, 5))
    export_ax = export_fig.add_subplot(111)
    
    axis_mode = radio_axis.value_selected
    pos = slider_pos.val
    metric_mode, norm, levels, cmap, _, is_abs = get_visualization_params()
    
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
    
    values = calculate_metric(tpp, fnp, tpu, fnu, metric_mode, use_abs=is_abs)
    values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    values_masked = np.ma.masked_where(~mask_inside, values)
    
    img_exp = export_ax.imshow(values_masked, origin='lower', extent=[0,1,0,1], cmap=cmap, norm=norm)
    
    if np.any(~values_masked.mask):
        try:
            contours_exp = export_ax.contour(U, V, values_masked, levels=levels, 
                                             colors='white', linewidths=1.5, alpha=0.9)
            export_ax.clabel(contours_exp, inline=True, fontsize=10, fmt='%1.1f', colors='white')
        except:
            pass
            
    export_ax.set_xlabel(xlabel, fontsize=11)
    export_ax.set_ylabel(ylabel, fontsize=11)
    export_ax.set_title(f"{metric_mode} (Slice {pos:.2f}) {'[ABS]' if is_abs else ''}", fontsize=12)
    
    export_fig.colorbar(img_exp, ax=export_ax, shrink=0.8)
    
    fname = generate_filename("Export_2D", file_format)
    export_fig.savefig(fname, bbox_inches='tight', pad_inches=0.05, format=file_format, dpi=dpi_val)
    plt.close(export_fig)
    print(f"Zapisano (Lekki plik): {fname}")

# Wrapper functions for buttons
def save_3d_pdf(event): export_3d('pdf')
def save_3d_png(event): export_3d('png')
def save_2d_pdf(event): export_2d('pdf')
def save_2d_png(event): export_2d('png')

# -------------------------------------------------
#   Widgety
# -------------------------------------------------
bg_color = '#f0f0f0'

ax_radio_axis = plt.axes([0.05, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_axis = RadioButtons(ax_radio_axis, ['Z (TPp-TPu)', 'Y (TPp-FNp)', 'X (TPp-FNu)'], active=0)
ax_radio_axis.set_title("Oś Cięcia", fontsize=9)

ax_radio_metric = plt.axes([0.18, 0.02, 0.12, 0.18], facecolor=bg_color)
radio_metric = RadioButtons(ax_radio_metric, ['Eq. Opp. (Diff)', 'Eq. Opp. (Ratio)', 'Global Recall', 'Eq. Opp. (Norm)'], active=0)
ax_radio_metric.set_title("Metryka", fontsize=9)

ax_controls = plt.axes([0.31, 0.02, 0.16, 0.18], facecolor=bg_color)
ax_controls.axis('off')

# NOWY CHECKBOX - ABS VALUE
ax_check_abs = plt.axes([0.31, 0.205, 0.12, 0.05], frameon=False)
check_abs = CheckButtons(ax_check_abs, ['Abs Value'], [False])

ax_check_trans = plt.axes([0.31, 0.165, 0.12, 0.05], frameon=False)
check_transparency = CheckButtons(ax_check_trans, ['Transparent'], [True])

ax_check_plane = plt.axes([0.31, 0.125, 0.12, 0.05], frameon=False)
check_plane = CheckButtons(ax_check_plane, ['Show Plane'], [True])

ax_check_surface = plt.axes([0.31, 0.085, 0.12, 0.05], frameon=False)
check_surface = CheckButtons(ax_check_surface, ['Surface Only'], [False])

ax_check_log = plt.axes([0.31, 0.045, 0.12, 0.05], frameon=False)
check_log = CheckButtons(ax_check_log, ['Log Scale'], [False])

ax_text_log = plt.axes([0.43, 0.05, 0.04, 0.04])
text_log = TextBox(ax_text_log, '', initial="2")
text_log.label.set_size(9)

ax_box = plt.axes([0.32, 0.015, 0.05, 0.04])
text_box = TextBox(ax_box, 'Res: ', initial=str(current_N))
text_box.label.set_size(9)

ax_slider_pos = plt.axes([0.50, 0.12, 0.43, 0.03])
slider_pos = Slider(ax_slider_pos, 'Poz. Cięcia ', 0.0, 1.0, valinit=0.5, color='red')
slider_pos.label.set_size(9)

ax_slider_roll = plt.axes([0.50, 0.07, 0.43, 0.03])
slider_roll = Slider(ax_slider_roll, 'Roll (3D) ', -180, 180, valinit=56, color='orange')
slider_roll.label.set_size(9)

# --- 4 Buttons for Save ---
ax_btn_3d_pdf = plt.axes([0.66, 0.015, 0.06, 0.04])
btn_3d_pdf = Button(ax_btn_3d_pdf, '3D PDF', color='lightblue', hovercolor='0.9')

ax_btn_3d_png = plt.axes([0.73, 0.015, 0.06, 0.04])
btn_3d_png = Button(ax_btn_3d_png, '3D PNG', color='lightblue', hovercolor='0.9')

ax_btn_2d_pdf = plt.axes([0.82, 0.015, 0.06, 0.04])
btn_2d_pdf = Button(ax_btn_2d_pdf, '2D PDF', color='lightgreen', hovercolor='0.9')

ax_btn_2d_png = plt.axes([0.89, 0.015, 0.06, 0.04])
btn_2d_png = Button(ax_btn_2d_png, '2D PNG', color='lightgreen', hovercolor='0.9')

# Callbacks
radio_axis.on_clicked(update)
radio_metric.on_clicked(update)
slider_pos.on_changed(update)
slider_roll.on_changed(update_roll)
check_transparency.on_clicked(update)
check_plane.on_clicked(update)
check_log.on_clicked(update)
text_log.on_submit(update)
check_abs.on_clicked(update)

check_surface.on_clicked(update_point_cloud_data)
text_box.on_submit(update_point_cloud_data)

btn_3d_pdf.on_clicked(save_3d_pdf)
btn_3d_png.on_clicked(save_3d_png)
btn_2d_pdf.on_clicked(save_2d_pdf)
btn_2d_png.on_clicked(save_2d_png)

update()
plt.show()