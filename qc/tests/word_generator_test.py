import unittest

from qc.word_generator import WordGenerator


class GenerateWordsTest(unittest.TestCase):
    def test_generate_words(self):
        gates = [['H', 'Z'], ['H', 'Z'], ['H', 'Z', 'Y']]
        k = [2, 1, 2]
        want = [['HH', 'HZ', 'ZH', 'ZZ'], ['H', 'Z'], ['HH', 'HZ', 'HY', 'ZH', 'YH', 'ZY', 'YZ', 'ZZ', 'YY']]
        for i in range(3):
            generator = WordGenerator(gates[i], k[i])
            got = generator.generate_words()
            self.assertEqual(set(want[i]), set(got))


if __name__ == '__main__':
    unittest.main()
