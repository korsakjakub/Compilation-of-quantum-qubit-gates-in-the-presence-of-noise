# The ProgramInput class gathers the structure and behaviour of the inputs expected in the linear programs.
# The most important part of the class is the input variable, which is a list of dicts.


import os
import pickle
from typing import List, Dict

import numpy as np
from qworder.word_generator import WordGenerator

from config import Config
from channel import Channel, Noise, affine_channel_distance


def get_key_elements(arr: List[Dict], key: str) -> List:
    return [element[key] for element in arr]


class WordDict(Dict):
    def __init__(self, w: str, m: np.ndarray):
        super().__init__(w=w, m=m)

    def __str__(self):
        return f'WordDict: (word: {self["w"]}, matrix: {self["m"]})'


class ProgramInput:
    gates = {
        'H': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        'X': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        'Y': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        'Z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        'S': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        'T': np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'R': np.array(
            [[np.cos(np.pi / 4), np.sin(np.pi / 4), 0], [- np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]]),
        'I': np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )
    }
    _matrix = np.eye(3)
    input: List[WordDict] = []

    def __init__(self, wg: WordGenerator, length: int, channel: Channel = None, input_list: List[WordDict] = None):
        self.path = Config.WORDS_DIR + "L" + str(length) + ".npy"
        if channel:
            self.channel = channel
        if wg:
            self.wg = wg
        if input_list:
            self.input = input_list
        else:
            self.input: List[WordDict] = []

    # Given a list of words_b,
    # it returns a dictionary,
    # where the keys are words_b
    # and values are computed matrix compositions
    def channels_from_words(self, words):
        mat = []
        for i in range(len(words)):
            g = self._combine(words[i])._matrix
            mat.append(g)
        matrices, indices = np.unique(mat, axis=0, return_index=True)
        for i in range(len(matrices)):
            self.input.append(WordDict(words[indices[i]], matrices[i]))
        return self

    def _get_universal(self, name: str):
        return self.gates[name]

    def _combine(self, word: List[str]):
        self._matrix = np.eye(3)
        for g in word:
            self._matrix = np.matmul(self._matrix, self._get_universal(g))
        return self

    def _remove_redundant(self):
        mat = get_key_elements(self.input, 'm')
        words = get_key_elements(self.input, 'w')
        matrices, indices = np.unique(mat, axis=0, return_index=True)
        self.input = []
        for i in range(len(matrices)):
            self.input.append(WordDict(words[indices[i]], matrices[i]))

    def remove_far_points(self, target, out_length: int = 500):
        self._remove_redundant()
        if self.input[0]['m'].shape == (3,):  # euclidean norm
            self.input = sorted(self.input, key=lambda vector: np.linalg.norm(vector['m'] - target))[0:out_length - 1]
            self.input.append(WordDict('I', np.zeros((3,), dtype=float)))
            return self
        elif self.input[0]['m'].shape == (3, 3):  # operator norm
            self.input = sorted(self.input, key=lambda vector: np.linalg.norm(vector['m'] - target, ord=2))[
                         0:out_length - 1]
            identity_check = any(['I' == el['w'] for el in self.input])
            if not identity_check:
                self.input.append(WordDict(w='I', m=np.zeros((3, 3), dtype=float)))
        elif self.input[0]['m'].shape == (4, 4):  # affine norm
            if target.shape == (3, 3):
                target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
                target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)
            self.input = sorted(self.input, key=lambda vector: affine_channel_distance(vector['m'], target))[
                         0:out_length - 1]
            identity_check = any(['I' == el['w'] for el in self.input])
            if not identity_check:
                self.input.append(WordDict(w='I', m=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])))
        return self

    def _write_states(self):
        save_file = open(self.path, "wb")
        words = self.wg.generate_words()  # chunk_size=4096)
        self.wg.cascader.rules.write_rules()
        mat = self.channels_from_words(words).input
        pickle.dump(mat, save_file)
        save_file.close()
        return self

    def get_vectors(self):
        if os.path.isfile(self.path):
            file = open(self.path, "rb")
            data = self.channel.add_noise(pickle.load(file), length=self.wg.length)
        else:
            data = self.channel.add_noise(self._write_states().input, length=self.wg.length)
        # Apply a vector to each matrix an return
        self.input = [WordDict(el['w'], np.dot(el['m'], np.array([0.0, 0.0, 1.0]))) for el in data]
        return self

    def get_channels(self):
        if os.path.isfile(self.path):
            file = open(self.path, "rb")
            data = pickle.load(file)
            self.input = self.channel.add_noise(data, self.wg.length)
            file.close()
        else:
            self.input = self.channel.add_noise(self._write_states().input, self.wg.length)
        return self


if __name__ == "__main__":
    for file in os.listdir(Config.WORDS_DIR):
        f = open(Config.WORDS_DIR + file, "rb")
        data = pickle.load(f)
        f.close()
        f = open(Config.WORDS_DIR + file, "wb")
        #print(data)
        data2 = [dict(x) for x in data]
        print(data2)

        pickle.dump(data2, f)
        f.close()
