import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, report):
        self.report = report

    def prepare_dataframe(self, x_name, y_name):
        x_vals = getattr(self.report.total, x_name)
        y_vals = getattr(self.report, y_name)

        x_vals = np.nan_to_num(x_vals, nan=0.0)
        y_vals = np.nan_to_num(y_vals, nan=0.0)

        if 'difference' in y_name:
            display_y_name = f"Fairness (1 - {y_name})"
            display_y_vals = 1 - np.abs(y_vals)
        else:
            display_y_name = y_name
            display_y_vals = y_vals

        return pd.DataFrame({
            x_name: x_vals,
            display_y_name: display_y_vals,
            f'norm_{x_name}': self._normalize(x_vals, x_name),
            f'norm_{display_y_name}': display_y_vals,
            f'raw_{y_name}': y_vals,
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
        if 'difference' in name:
            return 1 - np.abs(values)
        elif 'ratio' in name:
             with np.errstate(divide='ignore', invalid='ignore'):
                recip = np.divide(1.0, values, out=np.zeros_like(values), where=values!=0)
             return np.minimum(values, recip)
        return values