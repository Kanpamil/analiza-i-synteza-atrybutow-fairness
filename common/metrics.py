import numpy as np

def safe_div(n, d):
    """Bezpieczne dzielenie tablic numpy (zwraca NaN przy dzieleniu przez 0)."""
    return np.divide(n, d, out=np.full_like(d, np.nan, dtype=float), where=d!=0)

def get_base_metric(tp, fp, tn, fn, metric_name):
    """
    Oblicza podstawową metrykę (np. TPR, Accuracy, MCC) 
    na podstawie składowych macierzy pomyłek.
    """
    p = tp + fn
    n = fp + tn
    total = p + n
    pred_pos = tp + fp
    
    # Obsługa krótkich nazw dla wewnętrznych obliczeń oraz długich dla GUI
    if metric_name in ['TPR', 'Recall', 'TPR (Recall / Eq. Opp.)']:
        return safe_div(tp, p)
    elif metric_name in ['FPR', 'FPR (Predictive Equality)']:
        return safe_div(fp, n)
    elif metric_name in ['TNR', 'Specificity', 'TNR (Specificity)']:
        return safe_div(tn, n)
    elif metric_name in ['PPV', 'Precision', 'PPV (Precision / Pred. Parity)']:
        return safe_div(tp, pred_pos)
    elif metric_name in ['PR', 'Positive Rate (Demographic Parity)']:
        return safe_div(pred_pos, total)
    elif metric_name == 'Accuracy':
        return safe_div(tp + tn, total)
    elif metric_name == 'F1 Score':
        prec = safe_div(tp, pred_pos)
        rec = safe_div(tp, p)
        return safe_div(2 * prec * rec, prec + rec)
    elif metric_name == 'MCC':
        num = (tp * tn) - (fp * fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return safe_div(num, den)
        
    # Domyślnie zwracamy zera
    if isinstance(tp, np.ndarray):
        return np.zeros_like(tp, dtype=float)
    return 0.0

def calculate_fairness_analysis(inputs, category, metric_name, comparison_type, perf_scope):
    """
    Główna funkcja obliczająca finalną wartość do wizualizacji.
    """
    def get(k): return inputs[k]
    
    # Dane Protected
    tp_p, fp_p, tn_p, fn_p = get('TP_p'), get('FP_p'), get('TN_p'), get('FN_p')
    # Dane Unprotected
    tp_u, fp_u, tn_u, fn_u = get('TP_u'), get('FP_u'), get('TN_u'), get('FN_u')

    # 1. Kategoria: PERFORMANCE (Trafność)
    if category == 'Performance':
        if perf_scope == 'Protected (Prot)':
            return get_base_metric(tp_p, fp_p, tn_p, fn_p, metric_name)
        elif perf_scope == 'Unprotected (Unprot)':
            return get_base_metric(tp_u, fp_u, tn_u, fn_u, metric_name)
        elif perf_scope == 'Total (P+U)':
            return get_base_metric(tp_p+tp_u, fp_p+fp_u, tn_p+tn_u, fn_p+fn_u, metric_name)

    # 2. Kategoria: FAIRNESS (Sprawiedliwość)
    elif category == 'Fairness':
        
        # --- Obsługa Equalized Odds (Specjalny przypadek) ---
        # Equalized Odds wymaga sprawdzenia zarówno TPR jak i FPR
        if 'Equalized Odds' in metric_name:
            # Obliczamy TPR dla obu grup
            tpr_p = get_base_metric(tp_p, fp_p, tn_p, fn_p, 'TPR')
            tpr_u = get_base_metric(tp_u, fp_u, tn_u, fn_u, 'TPR')
            
            # Obliczamy FPR dla obu grup
            fpr_p = get_base_metric(tp_p, fp_p, tn_p, fn_p, 'FPR')
            fpr_u = get_base_metric(tp_u, fp_u, tn_u, fn_u, 'FPR')
            
            # Różnice bezwzględne
            diff_tpr = np.abs(tpr_p - tpr_u)
            diff_fpr = np.abs(fpr_p - fpr_u)
            
            if 'Max' in metric_name:
                # Equalized Odds (Max Diff)
                return np.maximum(diff_tpr, diff_fpr)
            else:
                # Equalized Odds (Avg Diff) - Domyślnie
                return 0.5 * (diff_tpr + diff_fpr)

        # --- Obsługa Standardowych Metryk (Diff / Ratio) ---
        else:
            val_p = get_base_metric(tp_p, fp_p, tn_p, fn_p, metric_name)
            val_u = get_base_metric(tp_u, fp_u, tn_u, fn_u, metric_name)
            
            if comparison_type == 'Difference (Diff)':
                return val_p - val_u
            elif comparison_type == 'Ratio (Iloraz)':
                return safe_div(val_p, val_u)

    # Fallback
    if isinstance(tp_p, np.ndarray):
        return np.zeros_like(tp_p, dtype=float)
    return 0.0