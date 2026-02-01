import itertools
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from common.metrics import FairReport, ClassReport

def generate_points_barycentric(n_vertices, resolution):
    points_barycentric = []
    for placements in itertools.combinations_with_replacement(range(resolution + 1), n_vertices - 1):
        padded = (0,) + placements + (resolution,)
        distances = np.diff(padded)
        points_barycentric.append(distances)

    return np.array(points_barycentric, dtype=np.float64) / resolution


class FairReportGenerator:
    def __init__(self, n_total=1000, p_ratio=0.5, base_rate_p=0.3, base_rate_u=0.3):
        self.n_total = n_total
        self.p_ratio = p_ratio
        
        cnt_p = int(n_total * p_ratio)
        cnt_u = n_total - cnt_p
        
        self.P_p = int(cnt_p * base_rate_p)
        self.N_p = cnt_p - self.P_p
        
        self.P_u = int(cnt_u * base_rate_u)
        self.N_u = cnt_u - self.P_u

    def generate_from_simplex(self, res=15, max_samples=None, jitter=0.01):
        weights_4d = generate_points_barycentric(4, res)
        
        corners = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [0.5, 0, 0.5, 0] 
        ])
        weights_4d = np.vstack([corners, weights_4d])

        weights_8d = []
        weights_8d = [np.concatenate([w_p, w_u]) for w_p, w_u in itertools.product(weights_4d, weights_4d)]
        
        pts8 = np.array(weights_8d)

        if max_samples is not None and max_samples < len(pts8):
            idx = np.random.choice(len(pts8), size=max_samples, replace=False)
            pts8 = pts8[idx]
            
            perfect_w = np.array([0.5, 0, 0.5, 0]) 
            perfect_8d = np.concatenate([perfect_w, perfect_w])
            worst_8d = np.array([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])
            pts8 = np.vstack([pts8, perfect_8d, worst_8d])

        if jitter > 0:
            noise = np.random.normal(0, jitter, pts8.shape)
            pts8_noisy = pts8 + noise
            pts8_noisy = np.clip(pts8_noisy, 0, 1)
            
            sum_p = pts8_noisy[:, :4].sum(axis=1, keepdims=True)
            pts8_noisy[:, :4] /= sum_p
            
            sum_u = pts8_noisy[:, 4:].sum(axis=1, keepdims=True)
            pts8_noisy[:, 4:] /= sum_u
            
            pts8 = pts8_noisy

        return self._apply_scaling_logic(pts8)

    def generate_random(self, n_samples=50):
        weights_prot = np.random.dirichlet((1, 1, 1, 1), n_samples)
        weights_unprot = np.random.dirichlet((1, 1, 1, 1), n_samples)
        pts8 = np.hstack([weights_prot, weights_unprot])
        return self._apply_scaling_logic(pts8)

    def _apply_scaling_logic(self, pts8):
        wp = pts8[:, :4] 
        wu = pts8[:, 4:] 

        def scale_vec(w, P, N):
            epsilon = 1e-12
            w = w + epsilon
            pos_sum = w[:, 0] + w[:, 3]
            neg_sum = w[:, 1] + w[:, 2]
            
            tp_float = (w[:, 0] / pos_sum) * P
            fn_float = (w[:, 3] / pos_sum) * P
            fp_float = (w[:, 1] / neg_sum) * N
            tn_float = (w[:, 2] / neg_sum) * N

            return np.round(tp_float), np.round(fp_float), np.round(tn_float), np.round(fn_float)

        tp_p, fp_p, tn_p, fn_p = scale_vec(wp, self.P_p, self.N_p)
        tp_u, fp_u, tn_u, fn_u = scale_vec(wu, self.P_u, self.N_u)

        prot = ClassReport(tp=tp_p, fp=fp_p, tn=tn_p, fn=fn_p)
        unprot = ClassReport(tp=tp_u, fp=fp_u, tn=tn_u, fn=fn_u)

        return FairReport(prot, unprot)