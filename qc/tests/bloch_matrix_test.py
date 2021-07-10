import unittest

from qc.bloch_matrix import *


class BlochMatrixTest(unittest.TestCase):
    def test_get_rotation_matrix(self):
        angle = (np.pi / 2, np.pi / 12, np.pi / 6)
        axis = (Axis.Z, Axis.X, Axis.Y)

        wanted = ([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                  [[1, 0, 0], [0, np.cos(angle[1]), -np.sin(angle[1])], [0, np.sin(angle[1]), np.cos(angle[1])]],
                  [[np.cos(angle[2]), 0, np.sin(angle[2])], [0, 1, 0], [-np.sin(angle[2]), 0, np.cos(angle[2])]])
        for i in range(2):
            got = get_rotation_matrix(axis[i], angle[i])
            np.testing.assert_array_almost_equal(got, wanted[i])

    def test_get_random_orthogonal(self):
        b = BlochMatrix()
        r = b.get_random()
        np.testing.assert_array_almost_equal(np.matmul(np.transpose(r.rot), r.rot), np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


if __name__ == '__main__':
    unittest.main()
