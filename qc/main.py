import concurrent.futures
import csv
from multiprocessing import Process

import matplotlib.pyplot as plt
import picos
import picos as pc
from timeit import default_timer as timer
from picos import Problem

from qc.bloch_matrix import *
from qc.word_generator import WordGenerator


def lp(min_length: int, max_length: int):
    rn0 = np.random.default_rng().normal(size=3)
    n0 = rn0 / np.linalg.norm(rn0)

    output_t = []
    output_length = []

    for length in range(min_length, max_length):
        wg = WordGenerator(['H', 'T', 'R'], length).generate_words_shorter_than()
        m = get_bloch_matrices(wg)
        v = get_bloch_vectors(m)

        problem = Problem()
        n = len(v)
        p = {}

        # dodaję zmienne
        for i in range(n - 1):
            p[i] = picos.RealVariable('p[{0}]'.format(i))
        t = picos.RealVariable('t')

        # każde p >= 0
        problem.add_list_of_constraints([p[i] >= 0 for i in range(n - 1)])
        # p sumują się do 1
        problem.add_constraint(1 == pc.sum([p[i] for i in range(n - 1)]))
        # wiąz na wektory
        problem.add_constraint(t * n0[0] == pc.sum([p[j] * v[j][0] for j in range(n - 1)]))
        problem.add_constraint(t * n0[1] == pc.sum([p[j] * v[j][1] for j in range(n - 1)]))
        problem.add_constraint(t * n0[2] == pc.sum([p[j] * v[j][2] for j in range(n - 1)]))

        problem.set_objective("max", t)
        solution = problem.solve(solver='mosek')
        output_t.append(float(t))
        output_length.append(length)
    return [output_length, output_t, n0]


def lp_for_threading(min_length=3, max_length=8, threads=2):
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(lp, min_length, max_length) for _ in range(threads)]

        for f in concurrent.futures.as_completed(results):
            results.append(f.result())

    # print(results[2:])
    return results[threads:]


def plot_output(ll, t, path: str = "out.png"):
    plt.plot(ll, t)
    plt.xlabel('L')
    plt.ylabel('t')
    plt.title('t(L)')
    plt.savefig(path)


def write_results(path: str, results: list, open_type: str = "w"):
    output_file = open(path, open_type)
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
        output_file.write(str(ll[0][i]) + ", " + str(mean_t[i]) + "\n")
    output_file.close()


def file_to_png(in_path: str = "out.txt", out_path: str = "out.png"):
    ll = []
    t = []
    with open(in_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ll.append(int(row[0]))
            t.append(float(row[1]))
    plot_output(ll, t, out_path)


if __name__ == "__main__":
    start = timer()
    res = lp_for_threading(9, 10, 12)
    write_results("out.txt", res)
    file_to_png()
    end = timer()
    print(f'czas: {end - start} s')
