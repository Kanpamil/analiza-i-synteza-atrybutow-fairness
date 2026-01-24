import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.metrics import ClassificationMetrics, FairnessMetrics
from core.simplex import generate_vertices_cartesian, to_cartesian, generate_samples
from processing.constants import ROUND_DECIMALS

@dataclass(frozen=True)
class SimulationData:
    metrics_class: type[ClassificationMetrics | FairnessMetrics]
    vertices_cartesian: np.ndarray
    vertex_labels: np.ndarray
    points_cartesian: np.ndarray
    points_barycentric: np.ndarray
    point_values: np.ndarray

    @classmethod
    def from_metric_name(cls, metric_name, resolution):
        if metric_name in ClassificationMetrics.metric_names():
            metrics_class = ClassificationMetrics
        elif metric_name in FairnessMetrics.metric_names():
            metrics_class = FairnessMetrics
        else:
            raise ValueError(f'Invalid metric name: {metric_name}')

        vertex_labels = metrics_class.component_names()
        n_vertices = len(vertex_labels)

        vertices_cartesian, points_cartesian, points_barycentric = generate_samples(n_vertices, resolution)

        metrics = metrics_class(*points_barycentric.T)
        point_values = metrics.calculate(metric_name)

        return cls(
            metrics_class,
            vertices_cartesian,
            vertex_labels,
            points_cartesian,
            points_barycentric,
            point_values
        )

    def round_points_barycentric(self):
        return SimulationData(
            self.metrics_class,
            self.vertices_cartesian.copy(),
            self.vertex_labels.copy(),
            self.points_cartesian.copy(),
            np.round(self.points_barycentric, ROUND_DECIMALS),
            self.point_values.copy()
        )

    def project_cartesian(self, indices, value_agg_func):
        vertices_cartesian_columns = [f'v_c{i}' for i in indices]

        v_df = pd.DataFrame(self.vertices_cartesian[:, indices], columns=vertices_cartesian_columns)
        v_df['vertex_labels'] = self.vertex_labels

        v_df = v_df.round({column: ROUND_DECIMALS for column in vertices_cartesian_columns})
        v_df = v_df.groupby(vertices_cartesian_columns, as_index=False, sort=False).agg({
            'vertex_labels': ', '.join
        })

        points_cartesian_columns = [f'p_c{i}' for i in indices]
        points_barycentric_columns = [f'p_b{i}' for i in range(self.points_barycentric.shape[1])]

        p_df = pd.DataFrame(self.points_cartesian[:, indices], columns=points_cartesian_columns)
        p_df[points_barycentric_columns] = self.points_barycentric
        p_df['point_values'] = self.point_values

        p_df = p_df.round({column: ROUND_DECIMALS for column in points_cartesian_columns})
        p_df = p_df.groupby(points_cartesian_columns, as_index=False, sort=False).agg({
            **{column: 'mean' for column in points_barycentric_columns},
            'point_values': value_agg_func
        })

        p_df = p_df.round({'point_values': ROUND_DECIMALS})
        p_df['point_values'] += 0.0

        return SimulationData(
            self.metrics_class,
            v_df[vertices_cartesian_columns].to_numpy(dtype=np.float64),
            v_df['vertex_labels'].to_numpy(),
            p_df[points_cartesian_columns].to_numpy(dtype=np.float64),
            p_df[points_barycentric_columns].to_numpy(dtype=np.float64),
            p_df['point_values'].to_numpy(dtype=np.float64)
        )

    def slice_barycentric(self, index_to_value):
        points_barycentric = self.points_barycentric
        point_values = self.point_values.copy()

        for index, value in index_to_value.items():
            mask = np.isclose(points_barycentric[:, index], value)

            points_barycentric = points_barycentric[mask]
            point_values = point_values[mask]

        indices_to_delete = list(index_to_value.keys())

        vertex_labels = np.delete(self.vertex_labels, indices_to_delete)
        vertices_cartesian = generate_vertices_cartesian(n_vertices=vertex_labels.shape[0])

        points_barycentric = np.delete(points_barycentric, indices_to_delete, axis=1)
        points_cartesian = to_cartesian(points_barycentric, vertices_cartesian)

        return SimulationData(
            self.metrics_class,
            vertices_cartesian,
            vertex_labels,
            points_cartesian,
            points_barycentric,
            point_values
        )

    def project_radviz(self, indices, value_agg_func):
        n_vertices = self.vertices_cartesian.shape[0]

        theta = 2 * np.pi / n_vertices
        offset = 2
        angles = (np.arange(n_vertices, dtype=np.float64) + offset) * theta

        anchors = np.column_stack([-np.cos(angles), np.sin(angles)])
        vertices_cartesian = np.zeros((n_vertices, 2), dtype=np.float64)
        vertices_cartesian[indices] = anchors

        points_cartesian = to_cartesian(self.points_barycentric, vertices_cartesian)

        points_cartesian_columns = [f'p_c{i}' for i in range(points_cartesian.shape[1])]
        points_barycentric_columns = [f'p_b{i}' for i in range(self.points_barycentric.shape[1])]

        df = pd.DataFrame(points_cartesian, columns=points_cartesian_columns)
        df[points_barycentric_columns] = self.points_barycentric
        df['point_values'] = self.point_values

        df = df.round({column: ROUND_DECIMALS for column in points_cartesian_columns})
        df = df.groupby(points_cartesian_columns, as_index=False, sort=False).agg({
            **{column: 'mean' for column in points_barycentric_columns},
            'point_values': value_agg_func
        })

        df = df.round({'point_values': ROUND_DECIMALS})
        df['point_values'] += 0.0

        return SimulationData(
            self.metrics_class,
            vertices_cartesian,
            self.vertex_labels.copy(),
            df[points_cartesian_columns].to_numpy(dtype=np.float64),
            df[points_barycentric_columns].to_numpy(dtype=np.float64),
            df['point_values'].to_numpy(dtype=np.float64)
        )
