import unittest

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator

from qc.channel import *


class BlochMatrixTest(unittest.TestCase):

    def test_unique(self):
        words = WordGenerator(['H', 'R', 'I', 'T', 'X', 'Y', 'Z'], 4, cascader=Cascader()).generate_words()
        bloch = BlochMatrix(noise=1.0)
        m = bloch.get_bloch_matrices(words)


if __name__ == '__main__':
    unittest.main()
