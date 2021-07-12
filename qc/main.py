import concurrent.futures
from timeit import default_timer as timer

import picos as pc
from picos import Problem
from qutip import rand_unitary
from tqdm import tqdm

from qc.bloch_matrix import *
from qc.data_manager import DataManager, StatesManager, remove_far_points
from qc.gates import Gate
from qc.word_generator import WordGenerator


class Program:

    def __init__(self, min_length: int = 3, max_length: int = 4):
        self.min_length = min_length
        self.max_length = max_length

    def perform_sdp(self, v):
        output_a = []
        output_length = []
        for length in tqdm(range(self.min_length, self.max_length)):
            problem = Problem()
            index = length - self.min_length
            quantum_states = v[index].states
            number_of_states = len(quantum_states)
            maximally_mixed_state = np.eye(2, dtype=complex) / 2
            # define rescaled quantum state
            rho = pc.HermitianVariable('rho', (2, 2))  # rho scaled with alpha
            problem.add_constraint(rho >> 0)
            visibility = pc.trace(rho)
            # IMPORTANT CONDITION
            problem.add_constraint(visibility <= 1)
            # define probabilities
            probabilities = [pc.RealVariable('p[{0}]'.format(i), lower=0) for i in
                             range(number_of_states)]
            problem.add_constraint(1 == pc.sum([probabilities[i] for i in range(number_of_states)]))
            # we want noisy quantum state to belong to convex hull of available states
            problem.add_constraint(
                rho + (1 - visibility) * maximally_mixed_state == pc.sum(
                    [probabilities[i] * quantum_states[i] for i in range(number_of_states)]))
            problem.set_objective("max", visibility)
            print(problem)
            problem.solve(solver='cvxopt')
            probabilities_values = [float(prob.value) for prob in probabilities]
            maximal_visibility = visibility.value.real
            state_solution = np.array(rho.value_as_matrix) / maximal_visibility
            test_sum = np.zeros((2, 2), dtype=complex)
            for index_state in range(number_of_states):
                test_sum += probabilities_values[index_state] * quantum_states[index_state]
            test_noisy_state = maximal_visibility * state_solution + (1 - maximal_visibility) * maximally_mixed_state
            print('maximal visibility:', maximal_visibility)
            print('for state:')
            print(test_noisy_state)
            print(test_sum)
            print(np.linalg.eigvalsh(test_noisy_state))
            print(np.trace(test_noisy_state))
            print(test_sum - test_noisy_state)
            # raise KeyError
            output_a.append(maximal_visibility)
            output_length.append(length)
        return [output_length, output_a]

    def perform_lp(self, v, n0):
        output_t = []
        output_length = []

        for length in tqdm(range(self.min_length, self.max_length)):

            problem = Problem()
            index = length - self.min_length
            p = {}
            # take only a specified number of input vectors
            vec = remove_far_points(v[index].states, target=n0, out_length=2000)
            n = len(vec)

            # dodaję zmienne
            for i in range(n):
                p[i] = pc.RealVariable('p[{0}]'.format(i))
            t = pc.RealVariable('t')

            # każde p >= 0
            problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
            # p sumują się do 1
            problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
            # wiąz na wektory
            problem.add_constraint(t * n0[0] == pc.sum([p[j] * vec[j][0] for j in range(n)]))
            problem.add_constraint(t * n0[1] == pc.sum([p[j] * vec[j][1] for j in range(n)]))
            problem.add_constraint(t * n0[2] == pc.sum([p[j] * vec[j][2] for j in range(n)]))

            problem.set_objective("max", t)
            problem.solve(solver='cvxopt')
            output_t.append(float(t))
            output_length.append(length)
        return [output_length, output_t, n0]

    def perform_lp_channels(self, v, n0):
        output_t = []
        output_length = []
        target = n0

        for length in tqdm(range(self.min_length, self.max_length)):

            problem = Problem()
            index = length - self.min_length
            p = {}
            # take only a specified number of input vectors
            # vec = remove_far_points(v[index].states, target=n0, out_length=500)
            vec = v[index].states
            n = len(vec)

            # dodaję zmienne
            for i in range(n):
                p[i] = pc.RealVariable('p[{0}]'.format(i))
            t = pc.RealVariable('t')

            # każde p >= 0
            problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
            # p sumują się do 1
            problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
            # wiąz na wektory
            problem.add_constraint(t * target.rot[0][0] == pc.sum([p[j] * vec[j][0][0] for j in range(n)]))
            problem.add_constraint(t * target.rot[0][1] == pc.sum([p[j] * vec[j][0][1] for j in range(n)]))
            problem.add_constraint(t * target.rot[0][2] == pc.sum([p[j] * vec[j][0][2] for j in range(n)]))
            problem.add_constraint(t * target.rot[1][0] == pc.sum([p[j] * vec[j][1][0] for j in range(n)]))
            problem.add_constraint(t * target.rot[1][1] == pc.sum([p[j] * vec[j][1][1] for j in range(n)]))
            problem.add_constraint(t * target.rot[1][2] == pc.sum([p[j] * vec[j][1][2] for j in range(n)]))
            problem.add_constraint(t * target.rot[2][0] == pc.sum([p[j] * vec[j][2][0] for j in range(n)]))
            problem.add_constraint(t * target.rot[2][1] == pc.sum([p[j] * vec[j][2][1] for j in range(n)]))
            problem.add_constraint(t * target.rot[2][2] == pc.sum([p[j] * vec[j][2][2] for j in range(n)]))

            problem.set_objective("max", t)
            problem.solve(solver='cvxopt')
            output_t.append(float(t))
            output_length.append(length)
        return [output_length, output_t, n0]

    def threaded_program(self, gates: list, bloch: BlochMatrix, gate: Gate, program: str, threads: int = 2):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
            target = []

            # generate random targets for each thread
            for _ in range(threads):
                if program == "lp":
                    rn0 = np.random.default_rng().normal(size=3)
                    target.append(rn0 / np.linalg.norm(rn0))
                elif program == "lp_channels":
                    target.append(bloch.get_random())
            # for each length generate input vectors - independent of target for now
            for length in tqdm(range(self.min_length, self.max_length)):
                wg = WordGenerator(gates, length)
                sm = StatesManager(bloch=bloch, gate=gate, wg=wg)
                if program == "lp":
                    v.append(sm.get_vectors())
                elif program == "sdp":
                    v.append(sm.get_states())
                elif program == "lp_channels":
                    v.append(sm.get_bloch_matrices())

            if program == "lp":
                results = [executor.submit(self.perform_lp, v, target[i]) for i in range(threads)]
            elif program == "sdp":
                results = [executor.submit(self.perform_sdp, v) for _ in range(threads)]
            elif program == "lp_channels":
                results = [executor.submit(self.perform_lp_channels, v, target[i]) for i in range(threads)]
            else:
                return None

            for f in concurrent.futures.as_completed(results):
                results.append(f.result())
        return results[threads:]


if __name__ == "__main__":
    gates = ['H', 'T', 'R', 'X', 'Y', 'Z', 'I']
    writer = DataManager()
    #for i in range(10):
    vis = 1.0 #round(1.0 - i/20, 2)
    #for _ in range(5):
    start = timer()
    program = Program(min_length=1, max_length=2)
    res = program.threaded_program(gates=gates, bloch=BlochMatrix(vis=vis), gate=Gate(vis=vis), program="sdp", threads=1)
    writer.write_results(res, vis)
    end = timer()
    print(f'czas: {end - start} s')
    # writer.file_to_png()
