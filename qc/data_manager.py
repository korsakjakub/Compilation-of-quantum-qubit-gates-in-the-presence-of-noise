import csv
import os.path

import numpy as np
from matplotlib import pyplot as plt

from qc.bloch_matrix import BlochMatrix, get_bloch_vectors
from qc.config import Config as cf
from qc.word_generator import WordGenerator


class DataManager:

    def __init__(self):
        self.txt = cf.TXT
        self.png = cf.PNG
        self.dir = cf.DIR

    def plot_output(self, ll, t) -> None:
        plt.plot(ll, t)
        plt.xlabel('L')
        plt.ylabel('t')
        plt.title('t(L)')
        plt.savefig(self.dir + self.png)

    def write_results(self, results: list, open_type: str = "w") -> None:
        output_file = open(self.dir + self.txt, open_type)
        t = []
        n0 = []
        ll = []
        for key in results:
            ll.append(key[0])
            t.append(key[1])
            n0.append(key[2])
        mean_t = []
        for el in np.transpose(t):
            mean_t.append(np.mean(el))

        for i in range(len(mean_t)):
            output_file.write(str(ll[0][i]) + "," + str(mean_t[i]) + "\n")
        output_file.close()

    def file_to_png(self) -> None:
        ll = []
        t = []
        with open(self.dir + self.txt) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                ll.append(int(row[0]))
                t.append(float(row[1]))
        self.plot_output(ll, t)


class StatesManager:

    def __init__(self, bloch: BlochMatrix, wg: WordGenerator):
        self.bloch = bloch
        self.wg = wg
        self.path = cf.DIR + str(wg.length) + "".join(wg.input_set) + ".txt"

    def _write_states(self) -> np.ndarray:
        output_file = open(self.path, "w")
        words = self.wg.generate_words_shorter_than()
        mat = self.bloch.get_bloch_matrices(words)
        vectors = get_bloch_vectors(mat)
        for vec in vectors:
            output_file.write(str(vec[0]) + "," + str(vec[1]) + "," + str(vec[2]) + "\n")
        output_file.close()
        print(vectors)
        return vectors

    def get_states(self) -> np.ndarray:
        if os.path.isfile(self.path):
            vectors = []
            with open(self.path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    vectors.append(np.array([float(row[0]), float(row[1]), float(row[2])]))
            return np.array(vectors)
        else:
            return self._write_states()


if __name__ == "__main__":
    sm = StatesManager(bloch=BlochMatrix(), wg=WordGenerator(['H', 'T', 'R'], 6))
    v = sm.get_states()
    print(v)
