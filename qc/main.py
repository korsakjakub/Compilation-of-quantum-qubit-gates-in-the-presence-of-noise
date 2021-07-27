import concurrent.futures
from time import sleep
from timeit import default_timer as timer

import picos as pc
from numpy import random
from picos import Problem

from qc.bloch_matrix import *
from qc.data_manager import DataManager, StatesManager, remove_far_points
from qc.gates import Gate
from qworder.word_generator import WordGenerator


class Program:

    def __init__(self, min_length: int = 3, max_length: int = 4):
        self.min_length = min_length
        self.max_length = max_length

    def perform_sdp(self, inputs):
        output_a = []
        output_length = []
        for length in tqdm(range(self.min_length, self.max_length)):
            problem = Problem()
            index = length - self.min_length
            quantum_states = inputs[index].states
            number_of_states = len(quantum_states)
            maximally_mixed_state = np.eye(2, dtype=complex) / 2.0001
            # define rescaled quantum state
            rho = pc.HermitianVariable('rho', (2, 2))  # rho scaled with alpha
            problem.add_constraint(rho >> 0)
            visibility = pc.trace(rho)
            # IMPORTANT CONDITION
            problem.add_constraint(visibility <= 1)
            # define probabilities
            probabilities = [pc.RealVariable('p[{0}]'.format(i)) for i in
                             range(number_of_states - 1)]
            problem.add_constraint(1 == pc.sum([probabilities[i] for i in range(number_of_states - 1)]))
            # we want noisy quantum state to belong to convex hull of available states
            problem.add_constraint(
                rho + (1 - visibility) * maximally_mixed_state == pc.sum(
                    [probabilities[i] * quantum_states[i] for i in range(number_of_states - 1)]))
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
        output_dist = []
        target = n0

        for length in tqdm(range(self.min_length, self.max_length)):

            problem = Problem()
            index = length - self.min_length
            p = {}
            # take only a specified number of input vectors
            vec = remove_far_points(np.concatenate([v[i].states for i in range(index, length)], axis=0),
                                    target=n0, out_length=200)
            #vec = np.concatenate([v[i].states for i in range(index, length)], axis=0)
            # vec = remove_far_points(v[index].states, target=n0, out_length=2000)
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

        #   print("\nprobs: ")
        #   print([float(p[i]) for i in range(len(p))])
        #   print("\ntarget: ")
        #   print(target.rot)
        #   print("\nout: ")
            out_rot = sum([float(p[i]) * vec[i] for i in range(n)])
        #   print(out_rot)
        #   print("output t: ")
        #   print(output_t)
        #   print("\ndistance: ")
            dist = np.linalg.norm(out_rot - target.rot, ord=np.inf)
            output_dist.append(dist)
        #   print(dist)
        return [output_length, output_t, n0, output_dist]

    def threaded_program(self, gates: list, bloch: BlochMatrix, gate: Gate, program: str, threads: int = 2):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
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

            return []
            results = []
            seed = random.randint(1, 1000000)
            # Generate target states for each thread
            if program == "lp":
                for _ in range(threads):
                    sleep(0.005)
                    seed += 1
                    random.seed(seed)
                    rn0 = np.random.default_rng().normal(size=3)
                    target = rn0 / np.linalg.norm(rn0)
                    results.append(executor.submit(self.perform_lp, v, target))
            elif program == "sdp":
                results = [executor.submit(self.perform_sdp, v) for _ in range(threads)]
            elif program == "lp_channels":
                for _ in range(threads):
                    sleep(0.5)
                    seed += 123
                    target = bloch.get_random(seed)
                    results.append(executor.submit(self.perform_lp_channels, v, target))
            else:
                return None

            for f in concurrent.futures.as_completed(results):
                try:
                    results.append(f.result())
                except ValueError:
                    print("cannot produce result")
        return results[threads:]


if __name__ == "__main__":
    gates = ['H', 'T', 'R', 'X', 'Y', 'Z', 'I']
    writer = DataManager()
    start = timer()

    # for v in tqdm(range(10)):
    vis = 1.0 # round(1.0 - v/20, 2)
    #for i in tqdm(range(30)):
        # print(str(i) + "-th iteration over " + str(vis))
    program = Program(min_length=7, max_length=8)
    res = program.threaded_program(gates=gates, bloch=BlochMatrix(vis=vis), gate=Gate(vis=vis), program="lp_channels",
                                   threads=1)
    writer.write_results(res, vis)
    end = timer()
    print(f'czas: {end - start} s')
    # writer.file_to_png()
