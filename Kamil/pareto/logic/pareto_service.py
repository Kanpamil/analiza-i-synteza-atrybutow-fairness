import numpy as np
import pandas as pd

class ParetoService:
    @staticmethod
    def identify_pareto(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
        """
        Identyfikuje punkty należące do frontu Pareto (Strict Pareto Front).
        Zakłada, że dla obu metryk WIĘCEJ = LEPIEJ.
        """
        # 1. Tworzymy kopię tylko potrzebnych danych
        subset = df[[x_col, y_col]].copy()
        
        # 2. Zmieniamy nazwy kolumn na BEZPIECZNE identyfikatory
        # Unikamy spacji (błąd atrybutu) ORAZ podkreślników na początku (błąd namedtuple)
        subset.columns = ['val_x', 'val_y']
        
        # Zapamiętujemy oryginalny indeks, żeby wiedzieć, który to był wiersz
        subset['orig_index'] = subset.index
        
        # 3. Sortowanie (Strict Pareto Logic)
        # - Najpierw malejąco po X (idziemy od prawej strony wykresu).
        # - Jeśli X są równe, malejąco po Y (bierzemy najlepszy Y dla danego X).
        subset = subset.sort_values(by=['val_x', 'val_y'], ascending=[False, False])
        
        # 4. Algorytm "Strict Cull"
        pareto_indices = []
        max_y_seen = -np.inf # Startujemy od minus nieskończoności
        
        for row in subset.itertuples():
            # Teraz bezpiecznie odwołujemy się do 'val_y'
            current_y = row.val_y
            
            # Warunek bycia na froncie:
            # Punkt musi być ŚCIŚLE lepszy na osi Y niż cokolwiek, 
            # co widzieliśmy do tej pory (dla punktów o lepszym lub równym X).
            if current_y > max_y_seen:
                pareto_indices.append(row.orig_index)
                max_y_seen = current_y
                
        # 5. Tworzymy wynikową maskę (True/False) dla oryginalnego DataFrame
        mask = pd.Series(False, index=df.index)
        mask.loc[pareto_indices] = True
        
        return mask

    @staticmethod
    def get_best_tradeoff_index(df, x_col, y_col):
        """
        Znajduje indeks punktu najbliższego ideałowi (1, 1).
        Przydatne do rekomendacji "najlepszego kompromisu".
        """
        # Obliczamy odległość euklidesową od punktu (1, 1)
        # Zakładamy, że kolumny są znormalizowane (0..1)
        dists = np.sqrt((1 - df[x_col])**2 + (1 - df[y_col])**2)
        return dists.idxmin()