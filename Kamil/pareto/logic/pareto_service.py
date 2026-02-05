import numpy as np
import pandas as pd

class ParetoService:
    @staticmethod
    def identify_pareto(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
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
        dists = np.sqrt((1 - df[x_col])**2 + (1 - df[y_col])**2)
        return dists.idxmin()

    @staticmethod
    def get_knee_point_index(df, x_col, y_col, pareto_mask):
        pareto_df = df[pareto_mask].copy()
        
        if len(pareto_df) < 3:
            return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)

        pareto_df = pareto_df.sort_values(by=x_col, ascending=False)
        
        x = pareto_df[x_col].values
        y = pareto_df[y_col].values
        
        x1, y1 = x[0], y[0]   
        x2, y2 = x[-1], y[-1] 
        
        denominator = x2 - x1
        if denominator == 0: 
            return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)
            
        a = (y2 - y1) / denominator
        b = y1 - a * x1
        
        y_line = a * x + b
        diffs = y - y_line
        
        best_idx_local = np.argmax(diffs)
        max_diff = diffs[best_idx_local]
        
        if max_diff <= 1e-9:
             return ParetoService.get_best_tradeoff_index(pareto_df, x_col, y_col)
        
        return pareto_df.index[best_idx_local]