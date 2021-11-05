import pickle
from typing import List

import numpy as np
from qworder.word_generator import WordGenerator

from qc.channel import Channel
from config import Config


class ProgramInput(object):

    gates = {
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

    def __init__(self, channel: Channel, wg: WordGenerator):
        self._matrix = np.eye(3)
        self.channel = channel
        self.wg = wg
        self.path = Config.WORDS_DIR + "L" + str(wg.length) \
                    + "".join(wg.input_set) + ".npy"
        self._states: list = []

    # Given a list of words,
    # it returns a dictionary,
    # where the keys are words
    # and values are computed matrix compositions
    def get_channels(self, words):
        out = {}
        mat = []
        for i in range(len(words)):
            g = self._combine(words[i])._matrix
            mat.append(g)
        matrices, indices = np.unique(mat, axis=0, return_index=True)
        for i in range(len(matrices)):
            out.update({words[indices[i]]: matrices[i]})
        return out

    def _get_universal(self, name: str):
        return self.gates[name]

    def _combine(self, word: List[str]):
        self._rot = np.identity(3)
        for g in word:
            self._rot = np.matmul(self._rot, self._get_universal(g))
        return self

    def remove_far_points(self, points, target, out_length: int = 500):
        output = []
        if points[0].shape == (3,):  # euclidean norm
            points = np.unique(points, axis=0)
            output = sorted(points, key=lambda vector: np.linalg.norm(vector - target))[0:out_length - 1]
            # d = [np.linalg.norm(output[i] - target) for i in range(len(output))]
            # print(d)
            output.append(np.array([0, 0, 0]))
            return output
        elif points[0].shape == (3, 3):  # operator norm
            points = np.unique(points, axis=0)
            output = sorted(points, key=lambda vector: np.linalg.norm(vector - target, ord=2))[0:out_length - 1]
            output.append(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        return output

    def _write_states(self, type: str) -> dict:
        save_file = open(self.path, "wb")
        words = self.wg.generate_words()
        self.wg.cascader.rules.write_rules()
        if type == "v":
            mat = self.get_channels(words)
            vectors = get_bloch_vectors(mat)
            pickle.dump(vectors, save_file)
            save_file.close()
            return vectors
        elif type == "b":
            mat = self.get_channels(words)
            pickle.dump(mat, save_file)
            save_file.close()
            return mat

    def get_vectors(self) -> np.ndarray:
        if os.path.isfile(self.path):
            file = open(self.path, "rb")
            # data = self.bloch.add_noise(np.load(self.path), length=self.wg.length)
            data = self.channel.add_noise(pickle.load(file), length=self.wg.length)
        else:
            data = self.channel.add_noise(self._write_states("b"), length=self.wg.length)
        t = []
        for d in data:
            t.append(np.array(d).dot([1, 0, 0]))
        self._states = np.array(t)
        return self._states

    def get_channels(self) -> dict:
        if os.path.isfile(self.path):
            file = open(self.path, "rb")
            self._states = self.channel.add_noise(pickle.load(file), self.wg.length)
        else:
            self._states = self.channel.add_noise(self._write_states("b"), self.wg.length)
        return self._states

    def get_vectors(self, input_channels, initial=np.array((0.0, 0.0, 1.0))) -> dict:
        vectors = {}
        for key in input_channels:
            vectors.update(np.dot(input_channels[key], initial))
        return vectors
