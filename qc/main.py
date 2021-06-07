import concurrent.futures
from timeit import default_timer as timer

import picos
import picos as pc
from picos import Problem
from tqdm import tqdm

from qc.bloch_matrix import *
from qc.data_manager import DataManager, StatesManager, remove_far_points
from qc.word_generator import WordGenerator


class Program:

    def __init__(self, min_length: int = 3, max_length: int = 4):
        if max_length > min_length >= 2:
            self.min_length = min_length
            self.max_length = max_length
        else:
            print("max length should be bigger than min length.")

    def perform_lp(self, v, n0):
        output_t = []
        output_length = []

        for length in tqdm(range(self.min_length, self.max_length)):

            problem = Problem()
            index = length - self.min_length
            p = {}
            # take only a specified number of input vectors
            vec = remove_far_points(v[index].states, target=n0, out_length=500)
            n = len(vec)

            # dodaję zmienne
            for i in range(n):
                p[i] = picos.RealVariable('p[{0}]'.format(i))
            t = picos.RealVariable('t')

            # każde p >= 0
            problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
            # p sumują się do 1
            problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
            # wiąz na wektory
            problem.add_constraint(t * n0[0] == pc.sum([p[j] * vec[j][0] for j in range(n)]))
            problem.add_constraint(t * n0[1] == pc.sum([p[j] * vec[j][1] for j in range(n)]))
            problem.add_constraint(t * n0[2] == pc.sum([p[j] * vec[j][2] for j in range(n)]))

            problem.set_objective("max", t)
            problem.solve(solver='mosek')
            output_t.append(float(t))
            output_length.append(length)
        return [output_length, output_t, n0]

    def threaded_lp(self, gates: list, bloch: BlochMatrix, threads: int = 2):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
            n0 = []

            # generate random targets for each thread
            for _ in range(threads):
                rn0 = np.random.default_rng().normal(size=3)
                n0.append(rn0 / np.linalg.norm(rn0))
            # for each length generate input vectors - independent of target for now
            for length in tqdm(range(self.min_length, self.max_length)):
                wg = WordGenerator(gates, length)
                sm = StatesManager(bloch=bloch, wg=wg)
                v.append(sm.get_states())

            results = [executor.submit(self.perform_lp, v, n0[i]) for i in range(threads)]

            for f in concurrent.futures.as_completed(results):
                results.append(f.result())
        return results[threads:]


if __name__ == "__main__":
    gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
    for _ in range(5):
        start = timer()
        program = Program(min_length=2, max_length=9)
        writer = DataManager()
        res = program.threaded_lp(gates=gates, bloch=BlochMatrix(visibility=0.95), threads=15)
        writer.write_results(res)
        writer.file_to_png()
        end = timer()
        print(f'czas: {end - start} s')
