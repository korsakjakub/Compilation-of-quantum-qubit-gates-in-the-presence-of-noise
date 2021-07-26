# Cascader class reduces sequences of quantum gates for given gate-set and generates
# a multiplication table of all possible reductions
import csv
import os

import numpy as np
from typing import Dict

from qworder import config
from qworder.word_generator import WordGenerator


class Cascader(object):
    base_gates = {
        'H': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        'X': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        'Y': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        'Z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        'T': np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'R': np.array(
            [[np.cos(np.pi / 4), np.sin(np.pi / 4), 0], [- np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'I': np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )
    }

    def __init__(self, wg: WordGenerator, rules_path: str = ""):
        self._rules_path = rules_path if rules_path else config.PATH
        self._wg = wg
        self._gateset = self._wg.input_set
        self.rules = self._load_existing_rules()

    def _load_existing_rules(self):
        rules = {}
        if os.path.isfile(self._rules_path):
            with open(self._rules_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    rules[row[0]] = row[1]
        return rules

    def cascade(self) -> Dict[int, list]:
        words = self._wg.get_words_dictionary()
        for length in words:
            # for word in words[length]:
            for k in range(len(words[length])):
                word = words[length][k]
                for i in range(len(word) - 1):
                    sub = word[i] + word[i + 1]
                    if sub in self.rules:
                        words[length][k] = word.replace(sub, self.rules[sub])
                        continue
                    replacement = self._check_product(word[i], word[i + 1])
                    if replacement:
                        word = word.replace(sub, replacement)
                        self.rules[sub] = replacement
                    else:
                        self.rules[sub] = sub
            words[length] = list(np.unique(words[length]))
        self._write_rules()
        return words

    def _check_product(self, a: str, b: str) -> str:
        if not len(self.base_gates[a]) or not len(self.base_gates[b]):
            return ""
        for letter in self.base_gates:
            bprod = np.matmul(self.base_gates[a], self.base_gates[b])
            bg = self.base_gates[letter]
            if np.allclose(bprod, bg):
                return letter
        return ""

    def _write_rules(self):
        output_file = open(self._rules_path, "w")
        for rule in self.rules:
            output_file.write(rule + "," + self.rules[rule] + "\n")
        output_file.close()


if __name__ == '__main__':
    cascader = Cascader(WordGenerator(['H', 'T', 'R', 'X', 'Y', 'Z', 'I'], 3))
    print(cascader.cascade())
