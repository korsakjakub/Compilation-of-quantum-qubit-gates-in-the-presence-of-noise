import concurrent.futures
import itertools
import time
from itertools import chain

import numpy as np
import picos as pc
from picos import Problem
from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from scipy.spatial.transform import Rotation

import channel
import qc
from channel import affine_channel_distance
from lp_input import WordDict

import matplotlib.pyplot as plt


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


def brute_channel_affine(v, target):
    output3 = sorted(v, key=lambda vector: channel.affine_channel_distance(vector['m'], target))[0]
    return channel.affine_channel_distance(output3['m'], target), output3


def lp3_channels_affine(v, target):
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
    #vec = [m[i][j] + c[i] for i in range(len(v)) for j in range(3)]

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target + y == pc.sum([p[j] * m[j] for j in range(n)]))
    problem.add_constraint(w == pc.sum([p[j] * c[j] for j in range(n)]))
    problem.add_constraint(pc.norm(y) <= 1-t)
    problem.add_constraint(pc.norm(w) <= 1-t)

    problem.set_objective("max", t)
    print("\nTarget: \n", target)
    print(problem)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)
    d = affine_channel_distance(target, output)

    return t, d, output, p


def lp3_channels(v, target):
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
    problem.add_constraint(pc.norm(y) <= 1-t)

    problem.set_objective("max", t)
    print("\nTarget: \n", target)
    print(problem)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d = np.linalg.norm(output - target, 2)
    return t, d, output, p


def lp3_states(v, target):
    p = {}
    problem = Problem()
    n = len(v)
    I = pc.Constant('I', value=np.eye(2, dtype=complex))
    X = pc.Constant('X', value=np.array([[0, 1], [1, 0]], dtype=complex))
    Y = pc.Constant('Y', value=np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = pc.Constant('Z', value=np.array([[1, 0], [0, -1]], dtype=complex))

    def vectors_to_states(vector):
        return WordDict(w=vector['w'], m=0.5*(I + vector['m'][0]*X + vector['m'][1]*Y + vector['m'][2]*Z))

    v = [vectors_to_states(u) for u in v]
    v.append(WordDict(w='I', m=np.eye(2, dtype=complex)/2))
    target = 0.5 * (I + target[0] * X + target[1] * Y + target[2] * Z)
    #target = pc.Constant('target', value=[[8.39e-01 - 1j*0.00e+00, -3.07e-01 + 1j*2.03e-01],
                                          #[-3.07e-01 - 1j*2.03e-01, 1.61e-01 - 1j*0.00e+00]])

    for i in range(n):
        p[i] = pc.RealVariable('p[{0}]'.format(i))

    t = pc.RealVariable('t')
    a = pc.RealVariable('a')
    b = pc.RealVariable('b')
    c = pc.RealVariable('c')
    d = pc.RealVariable('d')

    y = a*I + b*X + c*Y + d*Z

    problem.add_list_of_constraints([p[i] >= 0 for i in range(n)])
    problem.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
    problem.add_constraint(t * target + y == pc.sum([p[j] * v[j]['m'] for j in range(n)]))
    problem.add_constraint(t * target + y >> 0)
    problem.add_constraint(pc.trace(y) == 1 - t)
    problem.add_constraint(b**2 + c**2 + d**2 <= 1/4*(1-t)**2)

    problem.set_objective("max", t)
    print(problem)
    problem.solve(solver='cvxopt')
    output = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    x = target.value-output.value
    d = np.trace(np.sqrt(x*x.H)).real
    return t, d, output, p


def lp2_channels_affine(v, target):
    n = len(v)
    r = {}
    q = {}
    problem2 = Problem()
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)

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
    problem2.add_constraint(t2 * target[3][0] + pc.sum([q[j] * v[j]['m'][3][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[3][1] + pc.sum([q[j] * v[j]['m'][3][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[3][2] + pc.sum([q[j] * v[j]['m'][3][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][3][2] for j in range(n)]))

    problem2.set_objective("max", t2)
    problem2.solve(solver='cvxopt')

    output2 = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix2 = sum([float(q[i]) * v[i]['m'] for i in range(n)])
    d2 = np.linalg.norm(output2 - target, 2)
    return t2, d2, output2, mix2, q, r


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


# this isn't good, because we cannot optimize over both {p_i} and u...
def lp1_channels_affine(v, target):
    n = len(v)
    p = {}
    problem1 = Problem()
    target = np.concatenate((target, np.array([[0], [0], [0]])), axis=1)
    target = np.concatenate((target, np.array([[0, 0, 0, 1]])), axis=0)

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
    problem1.add_constraint(t1 * target[3][0] == pc.sum([p[j] * v[j]['m'][3][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[3][1] == pc.sum([p[j] * v[j]['m'][3][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[3][2] == pc.sum([p[j] * v[j]['m'][3][2] for j in range(n)]))
    problem1.set_objective("max", t1)
    problem1.solve(solver='cvxopt')

    output1 = sum([float(p[i]) * v[i]['m'] for i in range(n)])
    d1 = affine_channel_distance(output1, target)
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
    problem1.add_constraint(t1 * target[0] == pc.sum([p[j] * v[j]['m'][0] for j in range(n)]))
    problem1.add_constraint(t1 * target[1] == pc.sum([p[j] * v[j]['m'][1] for j in range(n)]))
    problem1.add_constraint(t1 * target[2] == pc.sum([p[j] * v[j]['m'][2] for j in range(n)]))

    problem1.set_objective("max", t1)
    problem1.solve(solver='cvxopt')

    output1 = sum([float(p[i]) * v[i]['m'] for i in range(n)])
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
    problem2.add_constraint(t2 * target[0] + pc.sum([q[j] * v[j]['m'][0] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][0] for j in range(n)]))
    problem2.add_constraint(t2 * target[1] + pc.sum([q[j] * v[j]['m'][1] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][1] for j in range(n)]))
    problem2.add_constraint(t2 * target[2] + pc.sum([q[j] * v[j]['m'][2] for j in range(n)]) == pc.sum(
        [r[j] * v[j]['m'][2] for j in range(n)]))

    problem2.set_objective("max", t2)
    problem2.solve(solver='cvxopt')
    output2 = sum([float(r[i]) * v[i]['m'] for i in range(n)])
    mix2 = sum([float(q[i]) * v[i]['m'] for i in range(n)])
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

    def perform_states(self, inputs):

        outs = []
        previous_out = []
        t1comb = []
        t2comb = []
        t3comb = []
        d1comb = []
        d2comb = []
        d3comb = []

        for length in range(self.min_length, self.max_length):
            print("depth: ", length)
            index = length - self.min_length
            input_list = list(chain.from_iterable([inputs[i] for i in range(index + 1)]))
            v = qc.lp_input.ProgramInput(wg=self.wg, length=length, input_list=input_list) \
                .remove_far_points(target=self.targets, out_length=300)

            self.targets = np.array([0.8086422, -0.56722564, -0.15605407])
            print("target vector: ", self.targets)

            # LP1
            t1, d1, output1, p = lp1_states(v.input, self.targets)

            # LP2
            t2, d2, output2, mix2, q, r = lp2_states(v.input, self.targets)

            # LP3
            t3, d3, output3, p = lp3_states(v.input, self.targets)

            print("visibilities (mix/opt/con): ", t1, "\t", t2, "\t", t3)
            print("distances (mix/opt/con): ", d1, "\t", d2, "\t", d3)

            t1comb.append(float(t1))
            t2comb.append(float(t2))
            t3comb.append(float(t3))

            d1comb.append(float(d1))
            d2comb.append(float(d2))
            d3comb.append(float(d3))

        rn = np.arange(0, len(t1comb), 1)
        plt.plot(rn, t1comb, rn, t2comb, rn, t3comb)
        plt.ylabel('vis')
        plt.show()

        rn = np.arange(0, len(d1comb), 1)
        plt.plot(rn, d1comb, rn, d2comb, rn, d3comb)
        plt.ylabel('distance')
        plt.show()
        # BRUTE
#           output3 = sorted(v, key=lambda vector: np.linalg.norm(vector - self.targets, 2))[0]
#           d3 = np.linalg.norm(output3 - self.targets, 2)
        return ""

    def perform_initial_calculations_channels(self, inputs):
        outs = []
        t1comb = []
        t2comb = []
        t3comb = []
        d1comb = []
        d2comb = []
        d3comb = []
        self.targets = np.array([[0.50607966, 0.26960331, 0.8192664], [-0.59451586, 0.79721188, 0.10490047],
                                 [-0.62484739, -0.54015486, 0.56373616]])

        for length in range(self.min_length, self.max_length):
            index = length - self.min_length
            input_list = list(chain.from_iterable([inputs[i] for i in range(index + 1)]))
            v = qc.lp_input.ProgramInput(wg=self.wg, length=length, input_list=input_list)\
                .remove_far_points(target=self.targets, out_length=300)
            if self.noise_type.name == "AmplitudeDamping":
                t1, d1, output1, p = lp1_channels_affine(v.input, self.targets)
                t2, d2, output2, mix2, q, r = lp2_channels_affine(v.input, self.targets)
                t3, d3, output3, p = lp3_channels_affine(v.input, self.targets)
            else:
                t1, d1, output1, p = lp1_channels(v.input, self.targets)
                t2, d2, output2, mix2, q, r = lp2_channels(v.input, self.targets)
                t3, d3, output3, p = lp3_channels(v.input, self.targets)

            print("visibilities (mix/opt/con): ", t1, "\t", t2, "\t", t3)
            print("distances (mix/opt/con): ", d1, "\t", d2, "\t", d3)

            t1comb.append(float(t1))
            t2comb.append(float(t2))
            t3comb.append(float(t3))

            d1comb.append(float(d1))
            d2comb.append(float(d2))
            d3comb.append(float(d3))

        rn = np.arange(0, len(t1comb), 1)
        plt.plot(rn, t1comb, rn, t2comb, rn, t3comb)
        plt.ylabel('vis')
        plt.show()

        rn = np.arange(0, len(d1comb), 1)
        plt.plot(rn, d1comb, rn, d2comb, rn, d3comb)
        plt.ylabel('distance')
        plt.show()
            #d3, output3 = brute_channel(v.input, self.targets)
            #out = [length, self.targets.tolist(), float(t1), d1, output1, float(t2), d2, output2,
                   #mix2.tolist(), d3, output3]
            #outs.append([v.input, p])
            #outs.append(out)
        #return outs
        return False

    def perform_n_times_lp_channels(self, new_words, number=5):
        def _cascade_list(words):
            cascader = Cascader()
            for i in range(len(words)):
                words[i] = cascader.cascade_word(words[i])
            return np.unique(words)

        for length in range(self.max_length, self.max_length + number):
            start_time_loop = time.perf_counter()
            new_words = _cascade_list(self.wg.add_layer(new_words))
            v_input = qc.lp_input.ProgramInput(wg=self.wg, length=length). \
                channels_from_words(new_words).remove_far_points(target=self.targets, out_length=100).input
            t1, d1, output1, p = lp1_channels(v_input, self.targets)
            print("distance: ", d1, "\tvisibility: ", t1, "\tlength: ", length, "\t#: ", len(v_input))
            end_time_loop = time.perf_counter()
            print(f"Execution Time (loop): {end_time_loop - start_time_loop:0.6f}")
            return t1, d1, output1, p, v_input

    def perform_leaving_only_best_channels(self, outs):

        v_input = outs[-1][0]
        p = outs[-1][1]
        print(len(p))
        p = [float(p[i]) for i in range(len(p))]
        (new_words, q) = leave_only_best(p, v_input)
        (t1, d1, output1, p, v_input) = self.perform_n_times_lp_channels(new_words)
        #       for length in range(self.max_length, self.max_length + 5):
        #           start_time_loop = time.perf_counter()
        #           new_words = cascade_list(self.wg.add_layer(new_words))
        #           v_input = v.channels_from_words(new_words).remove_far_points(target=target, out_length=100).input
        #           t1, d1, output1, p = lp1_channels(v_input, target)
        #           print("distance: ", d1, "\tvisibility: ", t1, "\tlength: ", length, "\t#: ", len(v_input))
        #           end_time_loop = time.perf_counter()
        #           print(f"Execution Time (loop): {end_time_loop - start_time_loop:0.6f}")
        return ['test']

    def perform_splitting_into_smaller_programs(self, list_of_inputs, chunk_size=100, processes=None):
        chunks = np.array_split(list_of_inputs, len(list_of_inputs) / chunk_size)

        # this doesn't work for some reason
        # with multiprocessing.Pool(processes) as workers:
        # chunk_results = workers.map(partial(lp1_channels, target=self.targets), chunks)
        # print(chunk_results)
        chunks_output = []
        for chunk in chunks:
            chunk = chunk.tolist()
            identity_check = any(['I' == el['w'] for el in chunk])
            if not identity_check:
                chunk.append(WordDict(w='I', m=np.zeros((3, 3), dtype=float)))
            (_, d1, output1, p) = lp1_channels(chunk, self.targets)
            chunks_output = sorted(list(itertools.chain(chunks_output, leave_only_best(batch_float_cast(p), chunk)[1])),
                                   key=lambda x: x[0], reverse=True)
        chunks_words = np.array(chunks_output, dtype=object).T[1].tolist()
        chunks_channels = qc.lp_input.ProgramInput(wg=self.wg, length=self.max_length) \
            .channels_from_words(chunks_words).input
        identity_check = any(['I' == el['w'] for el in chunks_channels])
        if not identity_check:
            chunks_channels.append(WordDict(w='I', m=np.zeros((3, 3), dtype=float)))
        (_, d1, output1, p) = lp1_channels(chunks_channels, self.targets)
        sorted_output = sorted(leave_only_best(batch_float_cast(p), chunks_channels)[1],
                               key=lambda x: x[0], reverse=True)
        print(d1)
        print(sorted_output)
        sorted_words = self.wg.add_layer(self.wg.add_layer(np.array(sorted_output, dtype=object).T[1].tolist()))
        sorted_output_channels = qc.lp_input.ProgramInput(wg=self.wg, length=self.max_length) \
            .channels_from_words(sorted_words).input
        return sorted_output_channels

    def threaded_program(self, channel: qc.channel.Channel, program: str, threads: int = 2):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
            for length in range(self.min_length, self.max_length):
                self.wg.length = length
                lpinput = qc.lp_input.ProgramInput(channel=channel, wg=self.wg, length=length)
                if program == "states":
                    v.append(lpinput.get_vectors().input.copy())
                elif program == "channels":
                    v.append(lpinput.get_channels().input.copy())
            results = []
            if program == "states":
                results.append(self.perform_states(v))
                #for i in range(threads):
                    #results.append(executor.submit(self.perform_states, v, self.targets[i]))
            elif program == "channels":
                results.append(self.perform_initial_calculations_channels(v))
            return results


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
                probabilities = [p/weights_sum for p in probabilities]
                print("new probs: ", probabilities)
            else:
                new_inputs.append(lp_input[i]['w'])
                new_probabilities.append((probabilities[i], new_inputs[-1]))
    new_probabilities = sorted(new_probabilities, key=lambda x: x[0], reverse=True)
    return new_inputs, new_probabilities


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

