import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, report):
        self.report = report

    def prepare_dataframe(self, x_name, y_name):
        # 1. Pobieramy wartości
        x_vals = getattr(self.report.total, x_name)
        y_vals = getattr(self.report, y_name)

        # (Opcjonalnie) Czyścimy NaN/Inf
        x_vals = np.nan_to_num(x_vals, nan=0.0)
        y_vals = np.nan_to_num(y_vals, nan=0.0)

        # --- NOWOŚĆ: Logika wyświetlania ---
        # Decydujemy, co pokazać na osi Y.
        # Jeśli metryka to "difference", chcemy pokazać "Fairness" (odwróconą).
        if 'difference' in y_name:
            # Tworzymy nową nazwę dla osi Y
            display_y_name = f"Fairness (1 - {y_name})"
            # Odwracamy wartości DO WYKRESU (nie tylko do Pareto)
            display_y_vals = 1 - np.abs(y_vals)
        else:
            # Dla innych metryk (np. ratio, accuracy) zostawiamy jak jest
            display_y_name = y_name
            display_y_vals = y_vals

        return pd.DataFrame({
            # --- OŚ X ---
            x_name: x_vals,
            
            # --- OŚ Y (To zmieniliśmy - teraz to jest "Fairness") ---
            display_y_name: display_y_vals,
            
            # --- KOLUMNY UKRYTE DLA PARETO ---
            # Pareto zawsze chce "im więcej tym lepiej".
            # Dla X (Accuracy) to po prostu kopia X.
            f'norm_{x_name}': self._normalize(x_vals, x_name),
            
            # Dla Y: Skoro display_y_vals jest już "im więcej tym lepiej", 
            # to norm_... jest po prostu jego kopią.
            f'norm_{display_y_name}': display_y_vals,
            
            # --- DANE ORYGINALNE (Dla tooltipa) ---
            # Warto zachować surową różnicę, żeby pokazać ją w dymku
            f'raw_{y_name}': y_vals,

            # --- LICZEBNOŚCI ---
            'TP_prot': np.round(self.report.prot.tp),
            'FP_prot': np.round(self.report.prot.fp),
            'TN_prot': np.round(self.report.prot.tn),
            'FN_prot': np.round(self.report.prot.fn),
            
            'TP_unp': np.round(self.report.unprot.tp),
            'FP_unp': np.round(self.report.unprot.fp),
            'TN_unp': np.round(self.report.unprot.tn),
            'FN_unp': np.round(self.report.unprot.fn)
        })

    def _normalize(self, values, name):
        """Pomocnicza funkcja normalizująca."""
        # Ta funkcja jest OK, ale teraz używamy jej głównie dla X
        # lub dla specyficznych metryk Y innych niż difference.
        if 'difference' in name:
            return 1 - np.abs(values)
        elif 'ratio' in name:
             with np.errstate(divide='ignore', invalid='ignore'):
                recip = np.divide(1.0, values, out=np.zeros_like(values), where=values!=0)
             return np.minimum(values, recip)
        return values