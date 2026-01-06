import utils; utils.fix_imports(); utils.suppress_streamlit_cache()
import unittest
import numpy as np
from parameterized import parameterized
from processing.simulation import SimulationData
from processing.dimensionality import slice_dimensions, squash_dimensions

class TestDimensionality(unittest.TestCase):
    def setUp(self):
        self.sim_data = SimulationData(
            report=None,
            verts=np.array([[ 1,  1,  1,  1,  1,  1,  1],
                            [-1,  1, -1,  1, -1,  1, -1],
                            [ 1, -1, -1,  1,  1, -1, -1],
                            [-1, -1,  1,  1, -1, -1,  1],
                            [ 1,  1,  1, -1, -1, -1, -1],
                            [-1,  1, -1, -1,  1, -1,  1],
                            [ 1, -1, -1, -1, -1,  1,  1],
                            [-1, -1,  1, -1,  1,  1, -1]]),
            labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            coords=np.array([[-1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
                             [ 0.0, -1.0,  0.0, -1.0,  0.0,  1.0,  0.0]]),
            weights=np.array([[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 1.0],
                              [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5, 0.5]]),
            colors=np.array([0.5, np.nan])
        )

    @parameterized.expand([
        ({'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}, 2),
        ({'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 1.0}, 0),
        ({'A': 0.0, 'B': 0.0, 'C': 0.0, 'H': 1.0}, 1)
    ])
    def test_slice_dimensions(self, contraint_to_value, n_left_points):
        sliced_sim_data = slice_dimensions(self.sim_data, contraint_to_value)

        self.assertEqual(sliced_sim_data.verts.shape, (4, 3))
        self.assertEqual(sliced_sim_data.labels, [l for l in self.sim_data.labels if l not in contraint_to_value.keys()])
        self.assertEqual(sliced_sim_data.coords.shape, (n_left_points, 3))
        self.assertEqual(sliced_sim_data.weights.shape, (n_left_points, 4))
        self.assertEqual(sliced_sim_data.colors.shape, (n_left_points,))

    @parameterized.expand([
        ([0, 1, 2], 'mean'),
        ([0, 1, 3], 'sum'),
        ([1, 2, 4], 'size')
    ])
    def test_squash_dimensions(self, dims, agg_func):
        squashed_sim_data = squash_dimensions(self.sim_data, dims, agg_func)

        self.assertEqual(squashed_sim_data.verts.shape, (8, 3))
        self.assertEqual(squashed_sim_data.labels, self.sim_data.labels)
        self.assertEqual(squashed_sim_data.coords.shape, (self.sim_data.coords.shape[0], 3))
        self.assertEqual(squashed_sim_data.weights.shape, (self.sim_data.coords.shape[0], 8))
        self.assertEqual(len(squashed_sim_data.colors), self.sim_data.coords.shape[0])

if __name__ == '__main__':
    unittest.main()
