import numpy as np
import pandas as pd

class ParetoService:
    @staticmethod
    def identify_pareto(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
        # --- (Twój dotychczasowy kod bez zmian) ---
        subset = df[[x_col, y_col]].copy()
        subset.columns = ['val_x', 'val_y']
        subset['orig_index'] = subset.index
        subset = subset.sort_values(by=['val_x', 'val_y'], ascending=[False, False])
        
        pareto_indices = []
        max_y_seen = -np.inf
        
        for row in subset.itertuples():
            current_y = row.val_y
            if current_y > max_y_seen:
                pareto_indices.append(row.orig_index)
                max_y_seen = current_y
                
        mask = pd.Series(False, index=df.index)
        mask.loc[pareto_indices] = True
        return mask

    @staticmethod
    def get_best_tradeoff_index(df, x_col, y_col):
        # --- (Metoda Utopii - bez zmian) ---
        dists = np.sqrt((1 - df[x_col])**2 + (1 - df[y_col])**2)
        return dists.idxmin()

    @staticmethod
    def get_knee_point_index(df, x_col, y_col, pareto_mask):
        """
        Znajduje Knee Point na surowych danych (bez normalizacji).
        Szuka punktu, który jest najbardziej "nad" linią łączącą skrajne punkty frontu.
        """
        # 1. Filtrujemy tylko punkty z frontu
        pareto_df = df[pareto_mask].copy()
        
        if len(pareto_df) < 3:
            return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)

        # 2. Sortujemy malejąco po X (Acc) - P1 to ten z prawej (Max Acc), P2 to ten z lewej
        pareto_df = pareto_df.sort_values(by=x_col, ascending=False)
        
        # 3. Pobieramy SUROWE wartości (Bez normalizacji)
        x = pareto_df[x_col].values
        y = pareto_df[y_col].values
        
        # 4. Wyznaczamy parametry prostej (cięciwy) łączącej P1 i P2
        x1, y1 = x[0], y[0]   # Skrajny prawy (Max X, Min Y)
        x2, y2 = x[-1], y[-1] # Skrajny lewy (Min X, Max Y)
        
        # Współczynnik kierunkowy prostej: a = (y2 - y1) / (x2 - x1)
        # Ponieważ sortowaliśmy malejąco, (x2 - x1) będzie ujemne.
        denominator = x2 - x1
        if denominator == 0: # Zabezpieczenie (pionowa linia)
            return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)
            
        a = (y2 - y1) / denominator
        b = y1 - a * x1
        
        # 5. Obliczamy "wypukłość" jako odległość w pionie od prostej
        # Równanie prostej: y_line = a*x + b
        # Różnica: y_rzeczywiste - y_line
        # Jeśli diff > 0 -> Punkt jest NAD linią (Wypukły - tego szukamy)
        # Jeśli diff < 0 -> Punkt jest POD linią (Wklęsły - odrzucamy)
        
        y_line = a * x + b
        diffs = y - y_line
        
        best_idx_local = np.argmax(diffs)
        max_diff = diffs[best_idx_local]
        
        # 6. Walidacja wklęsłości
        # Jeśli największa różnica jest <= 0, to znaczy, że żaden punkt nie wychodzi
        # nad cięciwę (front jest płaski lub wklęsły). 
        # Wtedy geometria Knee Point zawodzi -> zwracamy Utopię.
        if max_diff <= 1e-9:
             return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)
        
        return pareto_df.index[best_idx_local]