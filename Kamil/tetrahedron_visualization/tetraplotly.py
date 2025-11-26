import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------
#  Tetrahedron vertex coordinates
# -------------------------------------------------
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2, 0.0])
D = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])

# -------------------------------------------------
#  Function that maps barycentric weights → value
# -------------------------------------------------
def func_value(a,b,c,d):
    if a+b == 0:
        return np.nan
    val = (a+b) / (a+b+c+d)
    return val

# -------------------------------------------------
#  Convert barycentric to 3D point
# -------------------------------------------------
def f_point(a, b, c, d):
    w = np.array([a, b, c, d])
    if w.sum() == 0: 
        return np.zeros(3)
    w = w / w.sum()
    return w[0]*A + w[1]*B + w[2]*C + w[3]*D
# -------------------------------------------------
#  Filter points by barycentric equation
# -------------------------------------------------
def filter_barycentric(pts, vals, barycentric_coords, equation_func, target_value, tolerance=0.02):
    # Oblicz wartości równania dla wszystkich punktów
    equation_vals = []
    for i in range(len(barycentric_coords)):
        a, b, c, d = barycentric_coords[i]
        eq_val = equation_func(a, b, c, d)
        equation_vals.append(eq_val)
    
    equation_vals = np.array(equation_vals)
    
    # Znajdź punkty spełniające równanie
    mask = np.abs(equation_vals - target_value) <= tolerance
    mask = mask & ~np.isnan(equation_vals)  # usuń NaN
    
    print(f"Points matching equation: {mask.sum()} / {len(pts)}")
    
    return pts[mask], vals[mask], barycentric_coords[mask]

# -------------------------------------------------
#  Class ratio equation
# -------------------------------------------------
def class_ratio_eq(a, b, c, d):
    """Stosunek klas: (a+c)/(b+d)"""
    if b + d == 0:
        return np.nan
    return (a + c) / (b + d)

# -------------------------------------------------
#  Barycentric grid
# -------------------------------------------------
pts = []
vals = []
barycentric_coords = []  # for hover info
N = 80   # resolution

for i in range(N+1):
    for j in range(N+1-i):
        for k in range(N+1-i-j):
            l = N - i - j - k
            w = np.array([i,j,k,l]) / N
            pts.append(f_point(*w))
            vals.append(func_value(*w))
            barycentric_coords.append(w)
              
pts = np.array(pts)
vals = np.array(vals)
barycentric_coords = np.array(barycentric_coords)  # konwertuj na numpy array

# valid and invalid points
valid_mask = ~np.isnan(vals)
invalid_mask = np.isnan(vals)

print(f"Valid points: {valid_mask.sum()}")
print(f"Invalid points (NaN): {invalid_mask.sum()}")

# -------------------------------------------------
#  Plotly visualization
# -------------------------------------------------
fig = go.Figure()

# edges
edges = [(A,B), (A,C), (A,D), (B,C), (B,D), (C,D)]
for i, (P, Q) in enumerate(edges):
    fig.add_trace(go.Scatter3d(
        x=[P[0], Q[0]],
        y=[P[1], Q[1]],
        z=[P[2], Q[2]],
        mode='lines',
        line=dict(color='black', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
# valid points plot
if valid_mask.any():
    fig.add_trace(go.Scatter3d(
        x=pts[valid_mask, 0],
        y=pts[valid_mask, 1], 
        z=pts[valid_mask, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=vals[valid_mask],
            colorscale='Viridis',
            colorbar=dict(title="Function Value"),
            showscale=True
        ),
        customdata=barycentric_coords[valid_mask],  # przekaż współrzędne barycentryczne
        hovertemplate='<b>Valid Point</b><br>' +
                     'A: %{customdata[0]:.3f}<br>' +
                     'B: %{customdata[1]:.3f}<br>' +
                     'C: %{customdata[2]:.3f}<br>' +
                     'D: %{customdata[3]:.3f}<br>' +
                     'Value: %{marker.color:.3f}<extra></extra>'
    ))

# invalid points plot
if invalid_mask.any():
    fig.add_trace(go.Scatter3d(
        x=pts[invalid_mask, 0],
        y=pts[invalid_mask, 1],
        z=pts[invalid_mask, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red'
        ),
        customdata=barycentric_coords[invalid_mask],  # przekaż współrzędne barycentryczne
        hovertemplate='<b>Invalid Point</b><br>' +
                     'A: %{customdata[0]:.3f}<br>' +
                     'B: %{customdata[1]:.3f}<br>' +
                     'C: %{customdata[2]:.3f}<br>' +
                     'D: %{customdata[3]:.3f}<br>' +
                     'Value: NaN<extra></extra>'
    ))



# Etykiety wierzchołków
vertices = [('A', A), ('B', B), ('C', C), ('D', D)]
offset = 0.1  # dystans odsunięcia

for label, vertex in vertices:
    # Oblicz kierunek odsunięcia (od środka czworościanu)
    center = (A + B + C + D) / 4  # środek czworościanu
    direction = vertex - center   # wektor od środka do wierzchołka
    direction = direction / np.linalg.norm(direction)  # znormalizuj
    
    # Pozycja etykiety - odsunięta o offset w kierunku na zewnątrz
    label_position = vertex + offset * direction
    
    fig.add_trace(go.Scatter3d(
        x=[label_position[0]],
        y=[label_position[1]],
        z=[label_position[2]],
        mode='text',  # tylko tekst, bez markera
        text=[label],
        textfont=dict(size=20, color='red', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Layout
fig.update_layout(
    title='Tetrahedron Barycentric Visualization',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y', 
        zaxis_title='Z',
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),  # równomierny kąt widzenia
            projection=dict(type='orthographic')  # projekcja ortogonalna eliminuje zniekształcenia
        )
    ),
    width=1000,
    height=1000
)

filtered_pts, filtered_vals, filtered_coords = filter_barycentric(
    pts, vals, barycentric_coords,
    class_ratio_eq,    # funkcja równania (a+c)/(b+d)
    1/3,              # target value = 1/3
    tolerance=0.03    # tolerancja
)

fig.show()