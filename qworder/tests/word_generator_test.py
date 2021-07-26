import unittest

from qworder.word_generator import WordGenerator


class GenerateWordsTest(unittest.TestCase):
    def test_generate_words(self) -> None:
        gates = [['H', 'Z'], ['H', 'Z'], ['H', 'Z', 'Y']]
        k = [2, 1, 2]
        # want = [['HH', 'HZ', 'ZH', 'ZZ'], ['H', 'Z'], ['HH', 'HZ', 'HY', 'ZH', 'YH', 'ZY', 'YZ', 'ZZ', 'YY']]
        want = [['HZ', 'ZH'], ['H', 'Z'], ['HZ', 'HY', 'ZH', 'YH', 'ZY', 'YZ']]
        for i in range(3):
            generator = WordGenerator(gates[i], k[i])
            got = generator.generate_words()
            self.assertEqual(set(want[i]), set(got))

    def test_words(self) -> None:
        words = ['0', '1']
        # want = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100',
                # '1101', '1110', '1111']
        want = ['0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100',
                '1101', '1110']
        k = 4
        g = WordGenerator(words, k).generate_words()
        self.assertEqual(set(want), set(g))

    def test_generate_words_shorter_than(self) -> None:
        gates = ['a', 'n']
        k = 2
        want = ['n', 'na', 'a', 'an']

        generator = WordGenerator(gates, k)
        got = generator.generate_words_shorter_than()
        self.assertEqual(set(want), set(got))


if __name__ == '__main__':
    unittest.main()
