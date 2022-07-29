import multiprocessing
from itertools import chain

import multiprocessing
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import picos as pc
from picos import Problem
from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from channel import Channel, affine_channel_distance
from lp_input import WordDict, ProgramInput


def batch_float_cast(ps):
    out = []
    for p in ps.values():
        out.append(p.value)
    return out


def random_inputs(length, amount: int = 100):
    gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
    cascader = Cascader()
    out = [cascader.cascade_word("".join([gates[np.random.randint(0, 6)] for _ in range(length)])) for _ in
           range(amount)]
    out.append('I')
    return out


def leave_only_best(probabilities, lp_input):
    new_inputs = []
    new_probabilities = []
    for i in range(len(probabilities)):
        if probabilities[i] > 1e-5:
            if lp_input[i]['w'] == 'I':
                probabilities[i] = 0
                weights_sum = np.sum(probabilities)
                print("old probs: ", probabilities)
                probabilities = [p / weights_sum for p in probabilities]
                print("new probs: ", probabilities)
            else:
                new_inputs.append(lp_input[i]['w'])
                new_probabilities.append((probabilities[i], new_inputs[-1]))
    new_probabilities = sorted(new_probabilities, key=lambda x: x[0], reverse=True)
    return new_inputs, new_probabilities


def generate_target(amount):
    return [Rotation.random().as_matrix() for i in range(amount)]


def brute_channel(args):
    v = args[0]
    target = args[1]
    output = sorted(v, key=lambda vector: np.linalg.norm(vector['m'] - target, 2))[0]
    return round(np.linalg.norm(output['m'] - target, 2), 4)


def brute_channel_affine(args):
    v = args[0]
    target = args[1]
    if target.shape == (3, 3):
        target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
        target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)
    output = sorted(v, key=lambda vector: affine_channel_distance(vector['m'], target))[0]
    return round(affine_channel_distance(output['m'], target), 4)


def lp1_channels(args):
    v = args[0]
    target = args[1]
    n = len(v)
    p = {}
    problem = Problem()

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t = pc.RealVariable('t')

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target[0][0] == pc.sum([p[j] * v[j]['m'][0][0] for j in range(n)]))
    problem.add_constraint(t * target[0][1] == pc.sum([p[j] * v[j]['m'][0][1] for j in range(n)]))
    problem.add_constraint(t * target[0][2] == pc.sum([p[j] * v[j]['m'][0][2] for j in range(n)]))
    problem.add_constraint(t * target[1][0] == pc.sum([p[j] * v[j]['m'][1][0] for j in range(n)]))
    problem.add_constraint(t * target[1][1] == pc.sum([p[j] * v[j]['m'][1][1] for j in range(n)]))
    problem.add_constraint(t * target[1][2] == pc.sum([p[j] * v[j]['m'][1][2] for j in range(n)]))
    problem.add_constraint(t * target[2][0] == pc.sum([p[j] * v[j]['m'][2][0] for j in range(n)]))
    problem.add_constraint(t * target[2][1] == pc.sum([p[j] * v[j]['m'][2][1] for j in range(n)]))
    problem.add_constraint(t * target[2][2] == pc.sum([p[j] * v[j]['m'][2][2] for j in range(n)]))
    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')

    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d = round(np.linalg.norm(output - target, 2), 4)
    return d if d <= 1.0 else 1.0


def lp2_channels(args):
    v = args[0]
    target = args[1]
    n = len(v)
    r = {}
    q = {}
    problem = Problem()

    for i in range(n):
        r[i] = pc.RealVariable('p[{0}]'.format(i))

    for i in range(n):
        q[i] = pc.RealVariable('q[{0}]'.format(i))

    t = pc.RealVariable('t')

    problem.add_list_of_constraints([r[i] >= 0 for i in range(n)])
    problem.add_list_of_constraints([q[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([r[i] for i in range(n)]))
    problem.add_constraint(1 - t == pc.sum([q[i] for i in range(n)]))
    problem.add_constraint(t * target[0][0] + pc.sum([q[j] * v[j]['m'][0][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][0] for j in range(n)]))
    problem.add_constraint(t * target[0][1] + pc.sum([q[j] * v[j]['m'][0][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][1] for j in range(n)]))
    problem.add_constraint(t * target[0][2] + pc.sum([q[j] * v[j]['m'][0][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][2] for j in range(n)]))
    problem.add_constraint(t * target[1][0] + pc.sum([q[j] * v[j]['m'][1][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][0] for j in range(n)]))
    problem.add_constraint(t * target[1][1] + pc.sum([q[j] * v[j]['m'][1][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][1] for j in range(n)]))
    problem.add_constraint(t * target[1][2] + pc.sum([q[j] * v[j]['m'][1][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][2] for j in range(n)]))
    problem.add_constraint(t * target[2][0] + pc.sum([q[j] * v[j]['m'][2][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][0] for j in range(n)]))
    problem.add_constraint(t * target[2][1] + pc.sum([q[j] * v[j]['m'][2][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][1] for j in range(n)]))
    problem.add_constraint(t * target[2][2] + pc.sum([q[j] * v[j]['m'][2][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][2] for j in range(n)]))

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')

    output = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix = sum([float(q[i]) * v[i]['m'] for i in range(n)])
    d = round(np.linalg.norm(output - target, 2), 4)
    return d if d <= 1.0 else 1.0


def lp3_channels(args):
    v = args[0]
    target = args[1]
    p = {}
    problem = Problem()
    n = len(v)
    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t = pc.RealVariable('t')

    y = pc.RealVariable('Y', (3, 3))
    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target + y == pc.sum([p[j] * v[j]['m'] for j in range(n)]))
    problem.add_constraint(pc.SpectralNorm(y) <= 1 - t)
    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d = round(np.linalg.norm(output - target, 2), 4)
    return d if d <= 1.0 else 1.0


def lp1_channels_affine(args):
    v = args[0]
    target = args[1]
    n = len(v)
    p = {}
    problem = Problem()
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))

    t = pc.RealVariable('t')

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target[0][0] == pc.sum([p[j] * v[j]['m'][0][0] for j in range(n)]))
    problem.add_constraint(t * target[0][1] == pc.sum([p[j] * v[j]['m'][0][1] for j in range(n)]))
    problem.add_constraint(t * target[0][2] == pc.sum([p[j] * v[j]['m'][0][2] for j in range(n)]))
    problem.add_constraint(t * target[1][0] == pc.sum([p[j] * v[j]['m'][1][0] for j in range(n)]))
    problem.add_constraint(t * target[1][1] == pc.sum([p[j] * v[j]['m'][1][1] for j in range(n)]))
    problem.add_constraint(t * target[1][2] == pc.sum([p[j] * v[j]['m'][1][2] for j in range(n)]))
    problem.add_constraint(t * target[2][0] == pc.sum([p[j] * v[j]['m'][2][0] for j in range(n)]))
    problem.add_constraint(t * target[2][1] == pc.sum([p[j] * v[j]['m'][2][1] for j in range(n)]))
    problem.add_constraint(t * target[2][2] == pc.sum([p[j] * v[j]['m'][2][2] for j in range(n)]))
    problem.add_constraint(t * target[3][0] == pc.sum([p[j] * v[j]['m'][3][0] for j in range(n)]))
    problem.add_constraint(t * target[3][1] == pc.sum([p[j] * v[j]['m'][3][1] for j in range(n)]))
    problem.add_constraint(t * target[3][2] == pc.sum([p[j] * v[j]['m'][3][2] for j in range(n)]))
    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')

    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d = round(affine_channel_distance(output, target), 4)
    return d if d <= 1.0 else 1.0


def lp2_channels_affine(args):
    v = args[0]
    target = args[1]

    n = len(v)
    r = {}
    q = {}
    problem = Problem()
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)

    for i in range(n):
        r[i] = pc.RealVariable('p[{0}]'.format(i))

    for i in range(n):
        q[i] = pc.RealVariable('q[{0}]'.format(i))

    t = pc.RealVariable('t')

    problem.add_list_of_constraints([r[i] >= 0 for i in range(n)])
    problem.add_list_of_constraints([q[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([r[i] for i in range(n)]))
    problem.add_constraint(1 - t == pc.sum([q[i] for i in range(n)]))
    problem.add_constraint(t * target[0][0] + pc.sum([q[j] * v[j]['m'][0][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][0] for j in range(n)]))
    problem.add_constraint(t * target[0][1] + pc.sum([q[j] * v[j]['m'][0][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][1] for j in range(n)]))
    problem.add_constraint(t * target[0][2] + pc.sum([q[j] * v[j]['m'][0][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0][2] for j in range(n)]))
    problem.add_constraint(t * target[1][0] + pc.sum([q[j] * v[j]['m'][1][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][0] for j in range(n)]))
    problem.add_constraint(t * target[1][1] + pc.sum([q[j] * v[j]['m'][1][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][1] for j in range(n)]))
    problem.add_constraint(t * target[1][2] + pc.sum([q[j] * v[j]['m'][1][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1][2] for j in range(n)]))
    problem.add_constraint(t * target[2][0] + pc.sum([q[j] * v[j]['m'][2][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][0] for j in range(n)]))
    problem.add_constraint(t * target[2][1] + pc.sum([q[j] * v[j]['m'][2][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][1] for j in range(n)]))
    problem.add_constraint(t * target[2][2] + pc.sum([q[j] * v[j]['m'][2][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2][2] for j in range(n)]))
    problem.add_constraint(t * target[3][0] + pc.sum([q[j] * v[j]['m'][3][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][0] for j in range(n)]))
    problem.add_constraint(t * target[3][1] + pc.sum([q[j] * v[j]['m'][3][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][1] for j in range(n)]))
    problem.add_constraint(t * target[3][2] + pc.sum([q[j] * v[j]['m'][3][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][2] for j in range(n)]))

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')

    output = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix2 = sum([float(q[i]) * v[i]['m'] for i in range(n)])
    d = round(np.linalg.norm(output - target, 2), 4)
    return d if d <= 1.0 else 1.0


def lp3_channels_affine(args):
    v = args[0]
    target = args[1]
    p = {}
    problem = Problem()
    n = len(v)
    for i in range(3 * n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t = pc.RealVariable('t')
    y = pc.RealVariable('Y', (3, 3))
    w = pc.RealVariable('w', 3)

    m = [v[i]['m'][:3, :3] for i in range(len(v))]
    c = [v[i]['m'][3, :3] for i in range(len(v))]

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target + y == pc.sum([p[j] * m[j] for j in range(n)]))
    problem.add_constraint(w == pc.sum([p[j] * c[j] for j in range(n)]))
    problem.add_constraint(pc.SpectralNorm(y) <= 1 - t)
    problem.add_constraint(pc.Norm(w) <= 1 - t)

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)
    d = round(affine_channel_distance(target, output), 4)

    return d if d <= 1.0 else 1.0
    # return t, d, output, p


def lp1_states(args):
    v = args[0]
    target = args[1]
    n = len(v)
    p = {}
    problem = Problem()

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))
    t = pc.RealVariable('t')

    # każde p >= 0
    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    # p sumują się do 1
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    # wiąz na wektory
    problem.add_constraint(t * target[0] == pc.sum([p[j] * v[j]['m'][0] for j in range(n)]))
    problem.add_constraint(t * target[1] == pc.sum([p[j] * v[j]['m'][1] for j in range(n)]))
    problem.add_constraint(t * target[2] == pc.sum([p[j] * v[j]['m'][2] for j in range(n)]))

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')

    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d = np.linalg.norm(output - target, 2)
    return d


def lp2_states(args):
    v = args[0]
    target = args[1]
    r = {}
    q = {}
    problem = Problem()
    n = len(v)

    for i in range(n):
        r[i] = pc.RealVariable('r[{0}]'.format(i))

    for i in range(n):
        q[i] = pc.RealVariable('q[{0}]'.format(i))

    t = pc.RealVariable('t')

    # każde p >= 0
    problem.add_list_of_constraints([r[i] >= 0 for i in range(n)])
    problem.add_list_of_constraints([q[i] >= 0 for i in range(n)])
    # p sumują się do 1
    problem.add_constraint(1 == pc.sum([r[i] for i in range(n)]))
    problem.add_constraint(1 - t == pc.sum([q[i] for i in range(n)]))
    # wiąz na wektory
    problem.add_constraint(t * target[0] + pc.sum([q[j] * v[j]['m'][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0] for j in range(n)]))
    problem.add_constraint(t * target[1] + pc.sum([q[j] * v[j]['m'][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1] for j in range(n)]))
    problem.add_constraint(t * target[2] + pc.sum([q[j] * v[j]['m'][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2] for j in range(n)]))

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')
    output = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix = sum([float(q[i]) * v[i]['m'] for i in range(n)])
    d = np.linalg.norm(output - target, 2)
    return d


def lp3_states(args):
    v = args[0]
    target = args[1]
    p = {}
    problem = Problem()
    n = len(v)
    I = pc.Constant('I', value=np.eye(2, dtype=complex))
    X = pc.Constant('X', value=np.array([[0, 1], [1, 0]], dtype=complex))
    Y = pc.Constant('Y', value=np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = pc.Constant('Z', value=np.array([[1, 0], [0, -1]], dtype=complex))

    def vectors_to_states(vector):
        return WordDict(w=vector['w'], m=0.5 * (I + vector['m'][0] * X + vector['m'][1] * Y + vector['m'][2] * Z))

    v = [vectors_to_states(u) for u in v]
    v.append(WordDict(w='I', m=np.eye(2, dtype=complex) / 2))
    target = 0.5 * (I + target[0] * X + target[1] * Y + target[2] * Z)

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))

    t = pc.RealVariable('t')
    a = pc.RealVariable('a')
    b = pc.RealVariable('b')
    c = pc.RealVariable('c')
    d = pc.RealVariable('d')

    y = a * I + b * X + c * Y + d * Z

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target + y == pc.sum([p[j] * v[j]['m'] for j in range(n)]))
    problem.add_constraint(t * target + y >> 0)
    problem.add_constraint(pc.trace(y) == 1 - t)
    problem.add_constraint(b ** 2 + c ** 2 + d ** 2 <= 1 / 4 * (1 - t) ** 2)

    problem.set_objective("max", t)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    x = target.value - output.value
    d = np.trace(np.sqrt(x * x.H)).real
    return d


class Program:

    def __init__(self, wg: WordGenerator, min_length: int = 3, max_length: int = 4,
                 targets: np.ndarray = np.array([]), noise_type=""):
        self.wg = wg
        self.min_length = min_length
        self.max_length = max_length
        self.targets = targets
        self.noise_type = noise_type

    def perform_states(self, inputs):

        for length in range(1, self.max_length + 1):
            index = length - self.min_length + 1
            input_list = list(chain.from_iterable([inputs[i] for i in range(index)]))
            v = ProgramInput(wg=self.wg, length=length, input_list=input_list) \
                .remove_far_points(target=self.targets, out_length=300)

            self.targets = np.array([0.8086422, -0.56722564, -0.15605407])

            # LP1
            t1, d1, output1, p = lp1_states(v.input, self.targets)

            # LP2
            t2, d2, output2, mix2, q, r = lp2_states(v.input, self.targets)

            # LP3
            t3, d3, output3, p = lp3_states(v.input, self.targets)


            t1comb.append(float(t1))
            t2comb.append(float(t2))
            t3comb.append(float(t3))

            d1comb.append(float(d1))
            d2comb.append(float(d2))
            d3comb.append(float(d3))

        rn = np.arange(0, len(d1comb), 1)
        plt.plot(rn, d1comb, rn, d2comb, rn, d3comb)
        plt.ylabel('distance')
        plt.show()
        # BRUTE
        #           output3 = sorted(v, key=lambda vector: np.linalg.norm(vector - self.targets, 2))[0]
        #           d3 = np.linalg.norm(output3 - self.targets, 2)
        return ""

    def distribute_calculations_channels(self, channel: Channel, threads: int):
        inputs = []
        for length in range(1, self.max_length + 1):
            self.wg.length = length
            lpinput = ProgramInput(channel=channel, wg=self.wg, length=length)
            inputs.append(lpinput.get_channels().input.copy())

        outs = []
        # self.targets = np.array([[0.50607966, 0.26960331, 0.8192664], [-0.59451586, 0.79721188, 0.10490047],
        # [-0.62484739, -0.54015486, 0.56373616]])

        v_init = {i: [] for i in range(len(self.targets))}
        for length in tqdm(range(1, self.max_length + 1)):
            input_list = inputs[length - 1]
            for target_index in range(len(self.targets)):
                v = ProgramInput(wg=self.wg, length=length, input_list=input_list) \
                    .remove_far_points(target=self.targets[target_index], out_length=100)
                v_init[target_index] = list(chain.from_iterable([v_init[target_index], v.input]))

            if length < self.min_length:
                continue
            programs_input = [(v_init[ti], self.targets[ti]) for ti in range(len(self.targets))]
            with multiprocessing.Pool(threads) as workers:
                if self.noise_type.name == "AmplitudeDamping":
                    d1 = workers.map(lp1_channels_affine, programs_input)
                    d2 = workers.map(lp2_channels_affine, programs_input)
                    d3 = workers.map(lp3_channels_affine, programs_input)
                    d4 = workers.map(brute_channel_affine, programs_input)
                else:
                    d1 = workers.map(lp1_channels, programs_input)
                    d2 = workers.map(lp2_channels, programs_input)
                    d3 = workers.map(lp3_channels, programs_input)
                    d4 = workers.map(brute_channel, programs_input)

            out = [length, d1, d2, d3, d4]
            outs.append(out)
        return outs


if __name__ == "__main__":
    t1comb = [0.0, 0.24, 0.57, 0.69, 0.76, 0.85, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]
    t2comb = [0.0, 0.61, 0.80, 0.82, 0.83, 0.88, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]
    t3comb = [0.458, 0.69, 0.81, 0.83, 0.86, 0.92, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]
    d1comb = [1.0, 0.756, 0.428, 0.3, 0.23, 0.14, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    d2comb = [1.0, 0.695, 0.31, 0.3, 0.26, 0.13, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    d3comb = [0.92, 0.561, 0.32, 0.3, 0.24, 0.12, 0.08, 0.078, 0.078, 0.078, 0.078, 0.078]
    rn = np.arange(0, len(t1comb), 1)
    plt.plot(rn, t1comb, rn, t2comb, rn, t3comb)
    plt.ylabel('vis')
    plt.show()

    rn = np.arange(0, len(d1comb), 1)
    plt.plot(rn, d1comb, rn, d2comb, rn, d3comb)
    plt.ylabel('distance')
    plt.show()
