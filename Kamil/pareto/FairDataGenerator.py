import itertools
import numpy as np
import os
import sys

# Ustawienie ścieżek tak jak w oryginale
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from common.metrics import FairReport, ClassReport
from common.simplex_geometry import generate_simplex_grid

class FairReportGenerator:
    def __init__(self, n_total=1000, p_ratio=0.5, base_rate_p=0.3, base_rate_u=0.3):
        """
        Klasa zarządzająca generowaniem danych FairReport.
        """
        self.n_total = n_total
        self.p_ratio = p_ratio
        
        # Obliczenia liczebności grup (zgodnie z Twoją logiką)
        cnt_p = int(n_total * p_ratio)
        cnt_u = n_total - cnt_p
        
        self.P_p = int(cnt_p * base_rate_p)
        self.N_p = cnt_p - self.P_p
        
        self.P_u = int(cnt_u * base_rate_u)
        self.N_u = cnt_u - self.P_u

    def generate_from_simplex(self, res=15, max_samples=None, jitter=0.01):
        """
        Generuje dane systematyczne z opcjonalnym szumem (jitter), aby wygładzić pasy.
        """
        # 1. Generowanie siatki 4D (baza)
        _, weights_4d = generate_simplex_grid(4, res)
        
        # --- KROK NAPRAWCZY 1: Ręczne dodanie wierzchołków (Corner Cases) ---
        corners = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [0.5, 0, 0.5, 0] # Idealny balans
        ])
        weights_4d = np.vstack([corners, weights_4d])

        # 2. Iloczyn kartezjański 8D
        weights_8d = []
        for w_p, w_u in itertools.product(weights_4d, weights_4d):
            weights_8d.append(np.concatenate([w_p, w_u]))
        
        pts8 = np.array(weights_8d)

        # 3. Downsampling
        if max_samples is not None and max_samples < len(pts8):
            idx = np.random.choice(len(pts8), size=max_samples, replace=False)
            pts8 = pts8[idx]
            
            # Dodajemy punkty skrajne "na sztywno" po losowaniu
            perfect_w = np.array([0.5, 0, 0.5, 0]) 
            perfect_8d = np.concatenate([perfect_w, perfect_w])
            worst_8d = np.array([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])
            pts8 = np.vstack([pts8, perfect_8d, worst_8d])

        # --- KROK NAPRAWCZY 2: Jittering (Wygładzanie siatki) ---
        if jitter > 0:
            # Generujemy szum o rozkładzie normalnym
            noise = np.random.normal(0, jitter, pts8.shape)
            pts8_noisy = pts8 + noise
            
            # Musimy naprawić wagi po dodaniu szumu:
            # 1. Nie mogą być ujemne (clip)
            pts8_noisy = np.clip(pts8_noisy, 0, 1)
            
            # 2. Muszą sumować się do 1 w ramach grupy (4 pierwsze i 4 ostatnie)
            # Grupa Protected
            sum_p = pts8_noisy[:, :4].sum(axis=1, keepdims=True)
            pts8_noisy[:, :4] /= sum_p
            
            # Grupa Unprotected
            sum_u = pts8_noisy[:, 4:].sum(axis=1, keepdims=True)
            pts8_noisy[:, 4:] /= sum_u
            
            pts8 = pts8_noisy

        return self._apply_scaling_logic(pts8)

    def generate_random(self, n_samples=50):
        """Odpowiednik generate_random_tiny_data."""
        # KROK 1: Losowe wagi (Dirichlet)
        weights_prot = np.random.dirichlet((1, 1, 1, 1), n_samples)
        weights_unprot = np.random.dirichlet((1, 1, 1, 1), n_samples)
        
        pts8 = np.hstack([weights_prot, weights_unprot])
        
        return self._apply_scaling_logic(pts8)

    def _apply_scaling_logic(self, pts8):
        """
        Centralna logika przeliczania wag na liczebności z wektoryzacją.
        """
        wp = pts8[:, :4] 
        wu = pts8[:, 4:] 

        def scale_vec(w, P, N):
            # --- FIX: WYGŁADZANIE (EPSILON) ---
            # Dodajemy minimalną wartość do wag, aby uniknąć sytuacji,
            # gdzie suma (pos_sum lub neg_sum) wynosi 0.
            # Zapobiega to "znikaniu" ludzi, gdy generator wylosuje same zera dla danej klasy.
            epsilon = 1e-12
            w = w + epsilon

            # Sumy wag dla klas pozytywnej i negatywnej
            pos_sum = w[:, 0] + w[:, 3]
            neg_sum = w[:, 1] + w[:, 2]
            
            # Skalowanie (Teraz mianownik zawsze jest > 0)
            tp_float = (w[:, 0] / pos_sum) * P
            fn_float = (w[:, 3] / pos_sum) * P
            
            fp_float = (w[:, 1] / neg_sum) * N
            tn_float = (w[:, 2] / neg_sum) * N

            # Zaokrąglanie do liczb całkowitych ("Betonowanie" ludzi)
            return np.round(tp_float), np.round(fp_float), np.round(tn_float), np.round(fn_float)

        # Skalujemy
        tp_p, fp_p, tn_p, fn_p = scale_vec(wp, self.P_p, self.N_p)
        tp_u, fp_u, tn_u, fn_u = scale_vec(wu, self.P_u, self.N_u)

        # Tworzymy raporty
        prot = ClassReport(tp=tp_p, fp=fp_p, tn=tn_p, fn=fn_p)
        unprot = ClassReport(tp=tp_u, fp=fp_u, tn=tn_u, fn=fn_u)

        return FairReport(prot, unprot)