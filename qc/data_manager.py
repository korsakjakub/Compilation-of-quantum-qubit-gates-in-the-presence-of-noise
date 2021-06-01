import csv
import os.path

import numpy as np
from matplotlib import pyplot as plt

from qc.bloch_matrix import BlochMatrix, get_bloch_vectors
from qc.config import Config as cf
from qc.word_generator import WordGenerator

from datetime import datetime


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

    def write_results(self, results: list) -> None:
        t = []
        n0 = []
        ll = []
        for result in results:
            ll.append(result[0])
            t.append(result[1])
            n0.append(result[2])
        tt = np.transpose(t)
        for i in range(len(ll[0])):
            output_file = open(self.dir + str(ll[0][i]), "a")
            for j in range(len(tt[i])):
                output_file.write(str(tt[i][j]) + "\n")
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


class StatesManager:

    def __init__(self, bloch: BlochMatrix, wg: WordGenerator, target: np.ndarray = np.array([0, 0, 1])):
        self.bloch = bloch
        self.wg = wg
        self.path = cf.WORDS_DIR + "V" + str(self.bloch.visibility) + "L" + str(wg.length) \
                    + "".join(wg.input_set) + ".txt "
        self.target = target

    def write_states(self) -> np.ndarray:
        output_file = open(self.path, "a")
        words = self.wg.generate_words_shorter_than()
        mat = self.bloch.get_bloch_matrices(words)
        vectors = get_bloch_vectors(mat)
        for vec in vectors:
            output_file.write(str(vec[0]) + "," + str(vec[1]) + "," + str(vec[2]) + "\n")
        output_file.close()
        vectors = self.remove_far_points(vectors)
        return np.array(vectors)

    def get_states(self) -> np.ndarray:
        if os.path.isfile(self.path):
            vectors = []
            with open(self.path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    vectors.append(np.array([float(row[0]), float(row[1]), float(row[2])]))
            vectors = self.remove_far_points(vectors)
            return np.array(vectors)
        else:
            return self.write_states()

    def remove_far_points(self, vectors, threshold: float = np.sqrt(2)):
        out = []
        for vector in vectors:
            if np.linalg.norm(vector - self.target) < threshold:
                out.append(vector)
        return out


if __name__ == "__main__":
    sm = StatesManager(bloch=BlochMatrix(), wg=WordGenerator(['H', 'T', 'R'], 10))
    v = sm.get_states()
    dm = DataManager()
    dm.file_to_png()
