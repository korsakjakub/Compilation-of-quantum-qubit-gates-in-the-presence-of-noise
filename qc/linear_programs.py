import numpy as np
import scipy
from numpy import copy
from picos import Problem
import picos as pc
from scipy.spatial.transform import Rotation
import concurrent.futures
from itertools import chain
import qc
import time
from qworder.word_generator import WordGenerator
from qworder.cascading_rules import Cascader


def generate_target(program, amount):
    target = []
    if program == "states":
        rng = np.random.default_rng()
        for _ in range(amount):
            _rn0 = rng.normal(size=3)
            target.append(_rn0 / np.linalg.norm(_rn0))
    elif program == "channels":
        for _ in range(amount):
            target.append(Rotation.random().as_matrix())
    else:
        return False
    return target


def brute_channel(v, target):
    output3 = sorted(v, key=lambda vector: np.linalg.norm(vector['m'] - target, 2))[0]
    return np.linalg.norm(output3['m'] - target, 2), output3


def lp2_channels(v, target):
    n = len(v)
    r = {}
    q = {}
    problem2 = Problem()

    for i in range(n):
        r[i] = pc.RealVariable('p[{0}]'.format(i))

    for i in range(n):
        q[i] = pc.RealVariable('q[{0}]'.format(i))

    t2 = pc.RealVariable('t')

    problem2.add_list_of_constraints([r[i] >= 0 for i in range(n)])
    problem2.add_list_of_constraints([q[i] >= 0 for i in range(n)])
    problem2.add_constraint(1 == pc.sum([r[i] for i in range(n)]))
    problem2.add_constraint(1 - t2 == pc.sum([q[i] for i in range(n)]))
    problem2.add_constraint(t2 * target[0][0] + pc.sum([q[j] * v[j]['m'][0][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[0][1] + pc.sum([q[j] * v[j]['m'][0][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[0][2] + pc.sum([q[j] * v[j]['m'][0][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][2] for j in range(n)]))
    problem2.add_constraint(t2 * target[1][0] + pc.sum([q[j] * v[j]['m'][1][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[1][1] + pc.sum([q[j] * v[j]['m'][1][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[1][2] + pc.sum([q[j] * v[j]['m'][1][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][2] for j in range(n)]))
    problem2.add_constraint(t2 * target[2][0] + pc.sum([q[j] * v[j]['m'][2][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[2][1] + pc.sum([q[j] * v[j]['m'][2][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[2][2] + pc.sum([q[j] * v[j]['m'][2][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][2] for j in range(n)]))

    problem2.set_objective("max", t2)
    problem2.solve(solver='cvxopt')

    output2 = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix2 = sum([float(q[i]) * v[i]['m'] for i in range(n)])
    d2 = np.linalg.norm(output2 - target, 2)
    return t2, d2, output2, mix2, q, r


def lp1_channels(v, target):
    n = len(v)
    p = {}
    problem1 = Problem()

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t1 = pc.RealVariable('t')

    problem1.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem1.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem1.add_constraint(t1 * target[0][0] == pc.sum([p[j] * v[j]['m'][0][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[0][1] == pc.sum([p[j] * v[j]['m'][0][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[0][2] == pc.sum([p[j] * v[j]['m'][0][2] for j in range(n)]))
    problem1.add_constraint(t1 * target[1][0] == pc.sum([p[j] * v[j]['m'][1][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[1][1] == pc.sum([p[j] * v[j]['m'][1][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[1][2] == pc.sum([p[j] * v[j]['m'][1][2] for j in range(n)]))
    problem1.add_constraint(t1 * target[2][0] == pc.sum([p[j] * v[j]['m'][2][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[2][1] == pc.sum([p[j] * v[j]['m'][2][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[2][2] == pc.sum([p[j] * v[j]['m'][2][2] for j in range(n)]))
    problem1.set_objective("max", t1)
    problem1.solve(solver='cvxopt')

    output1 = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d1 = np.linalg.norm(output1 - target, 2)

    return t1, d1, output1, p


def lp1_states(v, target):
    n = len(v)
    p = {}
    problem1 = Problem()

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t1 = pc.RealVariable('t')

    # każde p >= 0
    problem1.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    # p sumują się do 1
    problem1.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    # wiąz na wektory
    problem1.add_constraint(t1 * target[0] == pc.sum([p[j] * v[j][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[1] == pc.sum([p[j] * v[j][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[2] == pc.sum([p[j] * v[j][2] for j in range(n)]))

    problem1.set_objective("max", t1)
    problem1.solve(solver='cvxopt')

    output1 = sum([float(p[i]) * v[i] for i in range(n)])
    d1 = np.linalg.norm(output1 - target, 2)
    return t1, d1, output1, p


def lp2_states(v, target):
    r = {}
    q = {}
    problem2 = Problem()
    n = len(v)

    for i in range(n):
        r[i] = pc.RealVariable('r[{0}]'.format(i))

    for i in range(n):
        q[i] = pc.RealVariable('q[{0}]'.format(i))

    t2 = pc.RealVariable('t')

    # każde p >= 0
    problem2.add_list_of_constraints([r[i] >= 0 for i in range(n)])
    problem2.add_list_of_constraints([q[i] >= 0 for i in range(n)])
    # p sumują się do 1
    problem2.add_constraint(1 == pc.sum([r[i] for i in range(n)]))
    problem2.add_constraint(1 - t2 == pc.sum([q[i] for i in range(n)]))
    # wiąz na wektory
    problem2.add_constraint(t2 * target[0] + pc.sum([q[j] * v[j][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[1] + pc.sum([q[j] * v[j][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[2] + pc.sum([q[j] * v[j][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j][2] for j in range(n)]))

    problem2.set_objective("max", t2)
    problem2.solve(solver='cvxopt')
    output2 = sum([float(r[i]) * v[i] for i in range(n)])
    mix2 = sum([float(q[i]) * v[i] for i in range(n)])
    d2 = np.linalg.norm(output2 - target, 2)
    return t2, d2, output2, mix2, q, r


class Program:

    def __init__(self, wg: WordGenerator, min_length: int = 3, max_length: int = 4,
                 targets: np.ndarray = np.array([]), noise_type=""):
        self.wg = wg
        self.min_length = min_length
        self.max_length = max_length
        self.targets = targets
        self.noise_type = noise_type

    def perform_states(self, inputs, target):

        outs = []
        previous_out = []

        for length in range(self.min_length, self.max_length):
            index = length - self.min_length
            v = np.concatenate([inputs[i] for i in range(index + 1)])
            hull = scipy.spatial.ConvexHull(v)
            simplices = hull.simplices
            points = []
            cm = []
            for simplex in simplices:
                points.append([])
                for s in simplex:
                    points[-1].append(v[s])
                cm.append(np.average(points[-1], axis=0))

            min = np.linalg.norm(cm[0] - target)
            min_index = 0
            for i in range(len(cm)):
                norm = np.linalg.norm(cm[i] - target)
                if norm < min:
                    min = norm
                    min_index = i

            v = points[min_index]
            v.append(np.array([0, 0, 0]))

            # LP1
            t1, d1, output1, p = lp1_states(v, target)

            # LP2
            t2, d2, output2, mix2, q, r = lp2_states(v, target)

            # BRUTE
            output3 = sorted(v, key=lambda vector: np.linalg.norm(vector - target, 2))[0]
            d3 = np.linalg.norm(output3 - target, 2)

            eps = 10e-5
            out = [length, target.tolist(), float(t1), d1, output1.tolist(), float(t2), d2, output2.tolist(),
                   mix2.tolist(), d3, output3.tolist(), hull.volume]
            if not previous_out:
                outs.append(copy.deepcopy(out))
                previous_out = copy.deepcopy(out)
                continue
            if previous_out[3] - d1 >= eps and previous_out[2] - float(t1) <= -eps:
                out1 = out[2:5]
                previous_out[2:5] = out[2:5]
            else:
                temp_out = copy.deepcopy(previous_out)
                out1 = temp_out[2:5]
            if previous_out[6] - d2 >= eps and previous_out[5] - float(t2) <= -eps:
                out2 = out[5:9]
                previous_out[5:9] = out[5:9]
            else:
                temp_out = copy.deepcopy(previous_out)
                out2 = temp_out[5:9]
            res_out = [length, target.tolist()]
            for e in out1:
                res_out.append(e)
            for e in out2:
                res_out.append(e)
            for e in out[9:]:
                res_out.append(e)
            outs.append(res_out)
        return outs

#   start_time = time.perf_counter()
#   print("Hello World")
#   end_time = time.perf_counter()

#   print(f"Start Time : {start_time}")
#   print(f"End Time : {end_time}")
#   print(f"Execution Time : {end_time - start_time:0.6f}")

    def perform_channels(self, inputs, target):
        outs = []
        v = p = q = r = None

        target = np.array([[0.50607966,  0.26960331,  0.8192664], [-0.59451586,  0.79721188,  0.10490047], [-0.62484739, -0.54015486,  0.56373616]])

        def leave_only_best(p, v_input):
            new_words = []
            q = []
            for i in range(len(p)):
                if p[i] > 1e-5:
                    new_words.append(v_input[i]['w'])
                    q.append((p[i], new_words[-1]))
            q = sorted(q, key=lambda x: x[0], reverse=True)
            return (new_words, q)

        def random_inputs(length, amount: int = 100):
            gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
            cascader = Cascader()
            out = [cascader.cascade_word("".join([gates[np.random.randint(0, 6)] for _ in range(length)])) for _ in range(amount)]
            out.append('I')
            return out

        def cascade_list(words):
            cascader = Cascader()
            for i in range(len(words)):
                words[i] = cascader.cascade_word(words[i])
            return np.unique(words)

        def perform_n(new_words, number=5):
            for length in range(self.max_length, self.max_length + number):
                start_time_loop = time.perf_counter()
                new_words = cascade_list(self.wg.add_layer(new_words))
                v_input = v.channels_from_words(new_words).remove_far_points(target=target, out_length=100).input
                t1, d1, output1, p = lp1_channels(v_input, target)
                print("distance: ", d1, "\tvisibility: ", t1, "\tlength: ", length, "\t#: ", len(v_input))
                end_time_loop = time.perf_counter()
                print(f"Execution Time (loop): {end_time_loop - start_time_loop:0.6f}")
                return t1, d1, output1, p, v_input

        for length in range(self.min_length, self.max_length):
            index = length - self.min_length
            #input_list = [inputs[i] for i in range(index + 1)]
            # newlist = list(chain.from_iterable(newlist))
            input_list = list(chain.from_iterable([inputs[i] for i in range(index + 1)]))
            v = qc.lp_input.ProgramInput(wg=self.wg, length=length, input_list=input_list) \
                .remove_far_points(target=target, out_length=300)
            t1, d1, output1, p = lp1_channels(v.input, target)
            t2, d2, output2, mix2, q, r = lp2_channels(v.input, target)
            d3, output3 = brute_channel(v.input, target)
            out = [length, target.tolist(), float(t1), d1, output1, float(t2), d2, output2,
                   mix2.tolist(), d3, output3]
            outs.append([v.input, p])
            print("distances: ", d1)

# obcinać od czasu do czasu

        v_input = outs[-1][0]
        p = outs[-1][1]
        print(len(p))
        p = [float(p[i]) for i in range(len(p))]
        (new_words, q) = leave_only_best(p, v_input)
        (t1, d1, output1, p, v_input) = perform_n(new_words)
#       for length in range(self.max_length, self.max_length + 5):
#           start_time_loop = time.perf_counter()
#           new_words = cascade_list(self.wg.add_layer(new_words))
#           v_input = v.channels_from_words(new_words).remove_far_points(target=target, out_length=100).input
#           t1, d1, output1, p = lp1_channels(v_input, target)
#           print("distance: ", d1, "\tvisibility: ", t1, "\tlength: ", length, "\t#: ", len(v_input))
#           end_time_loop = time.perf_counter()
#           print(f"Execution Time (loop): {end_time_loop - start_time_loop:0.6f}")
        return ['test']

    def threaded_program(self, channel: qc.channel.Channel, program: str, threads: int = 2):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
            for length in range(self.min_length, self.max_length):
                self.wg.length = length
                lpinput = qc.lp_input.ProgramInput(channel=channel, wg=self.wg, length=length)
                if program == "states":
                    v.append(lpinput.get_vectors())
                elif program == "channels":
                    v.append(lpinput.get_channels().input.copy())
            results = []
            if program == "states":
                for i in range(threads):
                    results.append(executor.submit(self.perform_states, v, self.targets[i]))
            elif program == "channels":
                results.append(self.perform_channels(v, self.targets[0]))
            return results


if __name__ == "__main__":
    def random_inputs(length, amount: int = 100):
        gates = ['H', 'T', 'S']
        out = ["".join([gates[np.random.randint(0, 6)] for _ in range(length)]) for _ in range(amount)]
        out.append('I')
        return out
    words = random_inputs(50, 2)
    print(words)
