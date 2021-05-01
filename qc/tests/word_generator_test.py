import unittest

from qc.word_generator import WordGenerator


class GenerateWordsTest(unittest.TestCase):
    def test_generate_words(self) -> None:
        gates = [['H', 'Z'], ['H', 'Z'], ['H', 'Z', 'Y']]
        k = [2, 1, 2]
        want = [['HH', 'HZ', 'ZH', 'ZZ'], ['H', 'Z'], ['HH', 'HZ', 'HY', 'ZH', 'YH', 'ZY', 'YZ', 'ZZ', 'YY']]
        for i in range(3):
            generator = WordGenerator(gates[i], k[i])
            got = generator.generate_words()
            self.assertEqual(set(want[i]), set(got))

    def test_words(self) -> None:
        slowa = ['0', '1']
        k = 4
        g = WordGenerator(slowa, k).generate_words()
        print(len(g), g)


if __name__ == '__main__':
    unittest.main()
