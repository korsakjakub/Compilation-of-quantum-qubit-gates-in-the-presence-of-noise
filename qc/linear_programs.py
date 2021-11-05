import concurrent

import numpy as np
import scipy
from numpy import copy
from picos import Problem
import picos as pc
from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from scipy.spatial.transform import Rotation

from qc.channel import Channel
from qc.results import StatesManager


class Program:

    def __init__(self, min_length: int = 3, max_length: int = 4, targets: np.ndarray = np.array([]), noise_type=""):
        self.min_length = min_length
        self.max_length = max_length
        self.targets = targets
        self.noise_type = noise_type

    def perform_states(self, input, target):

        outs = []
        previous_out = []

        # target = np.array([-0.5578337547883747, -0.6983624372220631, 0.4484544662459758])

        for length in range(self.min_length, self.max_length):
            index = length - self.min_length
            v = np.concatenate([input[i] for i in range(index + 1)])
            hull = scipy.spatial.ConvexHull(v)
            simplices = hull.simplices
            points = []
            cm = []
            for simplice in simplices:
                points.append([])
                for s in simplice:
                    points[-1].append(v[s])
                cm.append(np.average(points[-1], axis=0))

            min = np.linalg.norm(cm[0] - target)
            min_index = 0
            for i in range(len(cm)):
                norm = np.linalg.norm(cm[i] - target)
                if norm < min:
                    min = norm
                    min_index = i
            #           print("index: ", min_index)
            #           print("norm: ", min)
            #           print("best simplex: ", points[min_index])
            #           print("target: ", target)

            v = points[min_index]
            v.append(np.array([0, 0, 0]))
            # v = remove_far_points(np.concatenate([input[i] for i in range(index + 1)]),
            # target=target, out_length=1000)
            n = len(v)

            # LP1
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

            #           print(problem1)
            #           print("Target:")
            #           print(target)
            #           print("Weights:")
            #           print([float(p[i]) for i in range(n)])
            #           print("Input vectors:")
            #           print(v)
            output1 = sum([float(p[i]) * v[i] for i in range(n)])
            #           print("Output vector:")
            #           print(output1)
            d1 = np.linalg.norm(output1 - target, 2)

            print("d1: ", d1)
            print("t1: ", t1)

            # LP2
            r = {}
            q = {}
            problem2 = Problem()

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
            #           print(problem2)
            #           print("Target:")
            #           print(target)
            #           print("Weights r:")
            #           print([float(r[i]) for i in range(n)])
            #           print("Weights q:")
            #           print([float(q[i]) for i in range(n)])
            #           print("Input vector:")
            #           print(v)
            output2 = sum([float(r[i]) * v[i] for i in range(n)])
            #           print("Output vector:")
            #           print(output2)
            mix2 = sum([float(q[i]) * v[i] for i in range(n)])
            d2 = np.linalg.norm(output2 - target, 2)

            # BRUTE
            output3 = sorted(v, key=lambda vector: np.linalg.norm(vector - target, 2))[0]
            d3 = np.linalg.norm(output3 - target, 2)

            eps = 10e-5
            out = [length, target.tolist(), float(t1), d1, output1.tolist(), float(t2), d2, output2.tolist(),
                   mix2.tolist(), d3, output3.tolist(), hull.volume]
            #           print(str([length, float(t1), float(t2)]))
            #           print(str([length, d1, d2, d3]))
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
            # outs.append([length, target.tolist(), out1[:], out2[:], out[9:]])
            outs.append(res_out)
        return outs

    def perform_channels(self, input, target):

        outs = []
        previous_out = []
        print(input)

        for length in range(self.min_length, self.max_length):
            index = length - self.min_length

            v = remove_far_points(np.concatenate([input[i] for i in range(index + 1)]),
                                  target=target, out_length=100)
            # LP1
            t1, d1, output1, p = self.lp1_channels(v, target)
            # LP2
            t2, d2, output2, mix2, q, r = self.lp2_channels(v, target)
            # BRUTE
            d3, output3 = self.brute_channel(v, target)

            eps = 10e-5
            out = [length, target.tolist(), float(t1), d1, output1.tolist(), float(t2), d2, output2.tolist(),
                   mix2.tolist(), d3, output3.tolist()]

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

    def lp1_channels(self, v, target):
        n = len(v)
        p = {}
        problem1 = Problem()

        for i in range(n):
            p[i] = pc.RealVariable('p[{0}]'.format(i))
        t1 = pc.RealVariable('t')

        problem1.add_list_of_constraints([p[i] >= 0 for i in range(n)])
        problem1.add_constraint(1 == pc.sum([p[i] for i in range(n)]))
        problem1.add_constraint(t1 * target[0][0] == pc.sum([p[j] * v[j][0][0] for j in range(n)]))
        problem1.add_constraint(t1 * target[0][1] == pc.sum([p[j] * v[j][0][1] for j in range(n)]))
        problem1.add_constraint(t1 * target[0][2] == pc.sum([p[j] * v[j][0][2] for j in range(n)]))
        problem1.add_constraint(t1 * target[1][0] == pc.sum([p[j] * v[j][1][0] for j in range(n)]))
        problem1.add_constraint(t1 * target[1][1] == pc.sum([p[j] * v[j][1][1] for j in range(n)]))
        problem1.add_constraint(t1 * target[1][2] == pc.sum([p[j] * v[j][1][2] for j in range(n)]))
        problem1.add_constraint(t1 * target[2][0] == pc.sum([p[j] * v[j][2][0] for j in range(n)]))
        problem1.add_constraint(t1 * target[2][1] == pc.sum([p[j] * v[j][2][1] for j in range(n)]))
        problem1.add_constraint(t1 * target[2][2] == pc.sum([p[j] * v[j][2][2] for j in range(n)]))
        problem1.set_objective("max", t1)
        problem1.solve(solver='cvxopt')

        output1 = sum([float(p[i]) * v[i] for i in range(n)])
        d1 = np.linalg.norm(output1 - target, 2)

        m = 0
        for b in p:
            if b > 0:
                m += 1
        print(m)

        return t1, d1, output1, p

    def lp2_channels(self, v, target):
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
        problem2.add_constraint(t2 * target[0][0] + pc.sum([q[j] * v[j][0][0] for j in range(n)]) == pc.sum(
            [r[j] * v[j][0][0] for j in range(n)]))
        problem2.add_constraint(t2 * target[0][1] + pc.sum([q[j] * v[j][0][1] for j in range(n)]) == pc.sum(
            [r[j] * v[j][0][1] for j in range(n)]))
        problem2.add_constraint(t2 * target[0][2] + pc.sum([q[j] * v[j][0][2] for j in range(n)]) == pc.sum(
            [r[j] * v[j][0][2] for j in range(n)]))
        problem2.add_constraint(t2 * target[1][0] + pc.sum([q[j] * v[j][1][0] for j in range(n)]) == pc.sum(
            [r[j] * v[j][1][0] for j in range(n)]))
        problem2.add_constraint(t2 * target[1][1] + pc.sum([q[j] * v[j][1][1] for j in range(n)]) == pc.sum(
            [r[j] * v[j][1][1] for j in range(n)]))
        problem2.add_constraint(t2 * target[1][2] + pc.sum([q[j] * v[j][1][2] for j in range(n)]) == pc.sum(
            [r[j] * v[j][1][2] for j in range(n)]))
        problem2.add_constraint(t2 * target[2][0] + pc.sum([q[j] * v[j][2][0] for j in range(n)]) == pc.sum(
            [r[j] * v[j][2][0] for j in range(n)]))
        problem2.add_constraint(t2 * target[2][1] + pc.sum([q[j] * v[j][2][1] for j in range(n)]) == pc.sum(
            [r[j] * v[j][2][1] for j in range(n)]))
        problem2.add_constraint(t2 * target[2][2] + pc.sum([q[j] * v[j][2][2] for j in range(n)]) == pc.sum(
            [r[j] * v[j][2][2] for j in range(n)]))

        problem2.set_objective("max", t2)
        problem2.solve(solver='cvxopt')

        output2 = sum([float(r[i]) * v[i] for i in range(n)])
        mix2 = sum([float(q[i]) * v[i] for i in range(n)])
        d2 = np.linalg.norm(output2 - target, 2)
        return t2, d2, output2, mix2, q, r

    def brute_channel(self, v, target):
        output3 = sorted(v, key=lambda vector: np.linalg.norm(vector - target, 2))[0]
        d3 = np.linalg.norm(output3 - target, 2)
        return d3, output3

    def threaded_program(self, gates: list, channel: Channel, program: str, threads: int = 2):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            v = []
            # for each length generate input vectors - independent of target for now
            for length in range(self.min_length, self.max_length):
                wg = WordGenerator(gates, length, cascader=Cascader())
                sm = StatesManager(channel=channel, wg=wg)
                if program == "states":
                    v.append(sm.get_vectors())
                elif program == "channels":
                    v.append(sm.get_channels())

            results = []
            # Generate target states for each thread
            if program == "states":
                for i in range(threads):
                    results.append(executor.submit(self.perform_states, v, self.targets[i]))
            elif program == "channels":
                for i in range(threads):
                    results.append(executor.submit(self.perform_channels, v, self.targets[i]))
            else:
                return None
            for f in concurrent.futures.as_completed(results):
                try:
                    results.append(f.result())
                except ValueError:
                    print("cannot produce result")
        return results[threads:]

    def generate_target(self, program, amount):
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
