import unittest

import numpy as np

from qc.gates import Gate


class GatesTest(unittest.TestCase):
    def test_set_by_word(self):
        m = 'XX'
        want = np.array(((1, 0), (0, 1)))
        got = Gate().set_by_word(m).gate
        np.testing.assert_array_almost_equal(got, want)

    def test_set_by_params(self):
        a = (1 + 1j) / 2
        b = (1 - 1j) / 2
        g = Gate().set_by_params(a, b)
        got = g.gate
        want = np.array(((0.5 + 0.5j, -0.5 - 0.5j), (0.5 - 0.5j, 0.5 - 0.5j)))
        np.testing.assert_array_almost_equal(got, want)


if __name__ == '__main__':
    unittest.main()
