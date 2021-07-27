from __future__ import annotations

import csv
import os.path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from qc.bloch_matrix import BlochMatrix, get_bloch_vectors
from qc.config import Config as cf
from qc.gates import Gate
from qworder.word_generator import WordGenerator
from qworder.cascading_rules import Cascader
from qworder.rules import Word, Rules


class DataManager:

    def __init__(self):
        self.dir = cf.OUTPUTS_DIR
        self.fig_dir = cf.FIGURES_DIR

    def plot_output(self, ll, t) -> None:
        now = datetime.now()
        data = np.transpose(sorted(np.transpose([ll, t]), key=lambda x: x[0]))
        plt.plot(data[0], data[1])
        plt.xlabel('L')
        plt.ylabel('t')
        plt.title('t(L)')
        plt.savefig(self.fig_dir + now.strftime("%d-%m-%Y-%H%M%S") + ".png")

    def write_results(self, results: list, vis: float) -> None:
        t = []
        n0 = []
        ll = []
        d = []
        for result in results:
            ll.append(result[0])
            t.append(result[1])
            d.append(result[3])
        tt = np.transpose(t)
        dd = np.transpose(d)
        for i in range(len(ll[0])):
            output_file = open(self.dir + str(ll[0][i]) + "V" + str(vis), "a")
            for j in range(len(tt[i])):
                output_file.write(str(tt[i][j]) + "\t" + str(dd[i][j]) + "\n")
            output_file.close()

    def file_to_png(self) -> None:
        ll = []
        t = []
        for filename in os.listdir(cf.OUTPUTS_DIR):
            path = os.path.join(cf.OUTPUTS_DIR, filename)
            temp = []
            if os.path.isfile(path):
                with open(path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    ll.append(int(filename))
                    for row in csv_reader:
                        temp.append(float(row[0]))
                t.append(np.mean(temp))
        self.plot_output(ll, t)


def remove_far_points(points, target, out_length: int = 500):
    if points[0].shape == (3, 1):  # euclidean norm
        return sorted(points, key=lambda vector: np.linalg.norm(vector - target))[0:out_length]
    elif points[0].shape == (3, 3):  # operator norm
        return sorted(points, key=lambda vector: np.linalg.norm(vector - target.rot, ord=np.inf))[0:out_length]


class StatesManager(object):

    def __init__(self, bloch: BlochMatrix, gate: Gate, wg: WordGenerator):
        self.bloch = bloch
        self.gate = gate
        self.wg = wg
        self.path = cf.WORDS_DIR + "L" + str(wg.length) \
            + "".join(wg.input_set) + ".npy"
        self._states: list = []

    @property
    def states(self) -> list:
        return self._states

    def _write_states(self, type: str) -> np.ndarray:
        words = self.wg.generate_words()
        c = Cascader()
        for i in tqdm(range(len(words))):
            words[i] = c.cascade_word(Word(words[i], True)).word
        words = np.unique(words)
        if type == "v":
            mat = self.bloch.get_bloch_matrices(words)
            vectors = get_bloch_vectors(mat)
            np.save(self.path, vectors)
            return np.array(vectors)
        elif type == "m":
            mat = self.gate.get_gates(words)
            np.save(self.path, mat)
            return np.array(mat)
        elif type == "b":
            mat = self.bloch.get_bloch_matrices(words)
            np.save(self.path, mat)
            return np.array(mat)
        elif type == "s":
            rho0 = np.array([[1, 0], [0, 0]])
            mat = self.gate.get_gates(words)
            states = [mat[i] @ rho0 @ np.matrix.getH(mat[i]) for i in range(len(mat) - 1)]
            np.save(self.path, states)
            return np.array(states)

    def get_vectors(self) -> StatesManager:
        if os.path.isfile(self.path):
            data = np.load(self.path)
            self._states = np.array(data)
        else:
            self._states = self._write_states("v")
        return self

    def get_states(self) -> StatesManager:
        if os.path.isfile(self.path):
            matrices = []
            with open(self.path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    matrices.append(np.array([[float(row[0]), float(row[1])], [float(row[2]), float(row[3])]]))
                self._states = self.bloch.add_noise(np.array(matrices), self.wg.length)
        else:
            self._states = self.bloch.add_noise(self._write_states("s"), self.wg.length)
        return self

    def get_matrices(self) -> StatesManager:
        if os.path.isfile(self.path):
            self._states = np.load(self.path)
        else:
            self._states = self._write_states("m")
        return self

    def get_bloch_matrices(self) -> StatesManager:
        if os.path.isfile(self.path):
            self._states = self.bloch.add_noise(np.load(self.path), self.wg.length)
        else:
            self._states = self.bloch.add_noise(self._write_states("b"), self.wg.length)
        return self


if __name__ == "__main__":
    sm = StatesManager(bloch=BlochMatrix(), wg=WordGenerator(['H', 'T', 'R'], 2), gate=Gate())
    v = sm.get_matrices()
    print(v.states)
