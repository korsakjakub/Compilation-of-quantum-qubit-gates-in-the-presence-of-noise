import unittest

import numpy as np

from qc.density_matrix import DensityMatrix


class DensityMatrixTest(unittest.TestCase):
    def test_get_bloch_vector(self):
        mat = np.array(((0, 0), (0, 1)))
        want = np.array((0, 0, 1))
        rho = DensityMatrix(mat)
        got = rho.get_bloch_vector()
        self.assertEqual(got.all(), want.all())

    def test_get_rotation_matrix(self):
        pass

if __name__ == '__main__':
    unittest.main()
