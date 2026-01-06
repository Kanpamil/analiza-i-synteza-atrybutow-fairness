import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import Union
from core.metrics import ClassReport, FairReport
from core.simplex import generate_simplex_grid

@dataclass
class SimulationData:
    report: Union[ClassReport, FairReport]

    verts: np.ndarray
    labels: list[str]

    coords: np.ndarray
    weights: np.ndarray
    colors: np.ndarray

class SimulationFactory:
    @classmethod
    def create_simulation(cls, metric_name, res):
        if metric_name in ClassReport.available_metrics():
            n, labels = 4, ['TP', 'FP', 'TN', 'FN']
        elif metric_name in FairReport.available_metrics():
            n, labels = 8, ['TPp', 'FPp', 'TNp', 'FNp', 'TPup', 'FPup', 'TNup', 'FNup']
        else:
            raise ValueError(f'Invalid metric name: {metric_name}')

        report, verts, coords, weights = cls._get_simplex_data(n, res)
        colors = cls._get_colors(report, metric_name)

        return SimulationData(report, verts, labels, coords, weights, colors)

    @staticmethod
    @st.cache_data
    def _get_simplex_data(n, res):
        verts, coords, weights = generate_simplex_grid(n, res)

        if n == 4:
            report = ClassReport(*weights.T)
        elif n == 8:
            report = FairReport(ClassReport(*weights[:, :4].T), ClassReport(*weights[:, 4:].T))
        else:
            raise ValueError(f'Invalid number of vertices: {n}')

        return report, verts, coords, weights

    @staticmethod
    @st.cache_data
    def _get_colors(report, metric_name):
        return report.calculate(metric_name)
