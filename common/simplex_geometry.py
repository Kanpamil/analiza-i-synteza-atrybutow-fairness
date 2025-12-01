import numpy as np

# Definicje wierzchołków czworościanu foremnego
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 1.0])
C = np.array([1.0, 1.0, 0.0])
D = np.array([0.0, 1.0, 1.0])
STD_TETRA_VERTS = [A, B, C, D]

def get_simplex_vertices(n=4):
    """Zwraca współrzędne wierzchołków dla n-wymiarowego simpleksu w 3D."""
    if n == 1: return np.array(STD_TETRA_VERTS[0:1])
    elif n == 2: return np.array(STD_TETRA_VERTS[0:2])
    elif n == 3: return np.array(STD_TETRA_VERTS[0:3])
    elif n == 4: return np.array(STD_TETRA_VERTS[0:4])
    return np.array([])

def f_point(weights, vertices):
    """Mapuje wagi barycentryczne na punkt w przestrzeni 3D."""
    pos = np.zeros(3)
    for w, v in zip(weights, vertices):
        pos += w * v
    return pos

def generate_simplex_grid(n, N_res):
    """
    Generuje chmurę punktów (współrzędne 3D) oraz odpowiadające im wagi barycentryczne
    dla simpleksu o n wierzchołkach i rozdzielczości N_res.
    """
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

    return np.array(coords), np.array(weights_list)

def get_barycentric_for_slice(x, y, z):
    """
    Oblicza wagi barycentryczne dla punktu (x,y,z) wewnątrz czworościanu foremnego.
    Używane do generowania przekrojów 2D.
    """
    w_sum = (x + y + z) / 2.0
    w1 = 1.0 - w_sum
    w2 = w_sum - y
    w3 = w_sum - z
    w4 = w_sum - x
    return w1, w2, w3, w4