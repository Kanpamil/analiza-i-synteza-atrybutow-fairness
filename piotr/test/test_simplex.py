import unittest
import numpy as np
import scipy as sp
from parameterized import parameterized
from src.core.simplex import get_simplex_vertices, to_cartesian, to_barycentric, generate_simplex_grid

class TestSimplex(unittest.TestCase):
    @parameterized.expand([
        (2,),
        (4,),
        (8,)
    ])
    def test_simplex_vertices(self, n_points):
        verts = get_simplex_vertices(n_points)

        self.assertEqual(verts.shape, (n_points, n_points - 1))

        distances = sp.spatial.distance.pdist(verts)

        self.assertTrue(np.all(distances >= 0.5))
        self.assertTrue(np.allclose(distances, distances[0]))

    @parameterized.expand([
        (2, 'identity'),
        (4, 'identity'),
        (8, 'identity'),
        (2, 'random'),
        (4, 'random'),
        (8, 'random')
    ])
    def test_cartesian_barycentric_conversion(self, n_points, weights_type):
        verts = get_simplex_vertices(n_points)

        if weights_type == 'identity':
            weights = np.eye(n_points)
        else:
            np.random.seed(42)
            weights = np.random.dirichlet(np.ones(n_points), size=100)

        coords = to_cartesian(weights, verts)
        recovered_weights = to_barycentric(coords, verts)

        np.testing.assert_array_almost_equal(weights, recovered_weights)

    @parameterized.expand([
        (2, 4),
        (4, 8),
        (8, 16)
    ])
    def test_simplex_grid(self, n_points, res):
        _, coords, weights = generate_simplex_grid(n_points, res)

        self.assertEqual(coords.shape, (sp.special.comb(res + n_points - 1, n_points - 1, exact=True), n_points - 1))
        self.assertEqual(coords.shape, (weights.shape[0], weights.shape[1] - 1))

        self.assertTrue(np.allclose(np.sum(weights, axis=1), 1.0))
        self.assertTrue(np.all(weights >= 0.0))

if __name__ == '__main__':
    unittest.main()
