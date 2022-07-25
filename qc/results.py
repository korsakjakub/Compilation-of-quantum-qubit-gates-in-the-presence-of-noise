from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt

from qc.config import Config


class Results(object):

    def __init__(self):
        self.dir = Config.OUTPUTS_DIR
        self.fig_dir = Config.FIGURES_DIR

    def write(self, results: list, vis: float, program: str) -> None:
        for res in results:
            dir = self.dir + program + "/"
            path = dir + str(res[0]) + "V" + str(vis)
            if not os.path.exists(dir):
                os.makedirs(dir)
            output_file = open(path, "a")
            col_wout_l = res[1:]
            rows = np.array(col_wout_l).T.tolist()
            for row in rows:
                row_str = [str(el) for el in row]
                output_file.write('\t'.join(row_str) + '\n')
            output_file.close()

    def plot_depth(self, programs) -> None:
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.3, wspace=0)
        i = 0
        labels = ["No noise", "Depolarizing", "Amplitude damping"]
        for program in programs:
            path = self.dir + program + "/"
            for file in os.listdir(path):
                f = open(path + file, "r")
                _, t1, t2, t3, d1, d2, d3, d4 = f.read().split("\t")
                d1 = [float(item) for item in d1[1:len(d1)-2].split(", ")]
                d2 = [float(item) for item in d2[1:len(d2)-2].split(", ")]
                d3 = [float(item) for item in d3[1:len(d3)-2].split(", ")]
                d4 = [float(item) for item in d4[1:len(d4)-2].split(", ")]
                print(d1)
                print(d2)
                print(d3)
                print(d4)
                rn = np.arange(1, len(d1)+1, 1)
                ax[i].plot(rn, d1, label="Mixing")
                ax[i].plot(rn, d2, label="Optimized")
                ax[i].plot(rn, d3, label="Conic")
                ax[i].plot(rn, d4, label="Deterministic")
                ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
                ax[i].set_xlabel("Depth of compilation")
                break
            i += 1

        plt.xticks([3, 7, 12])
        #ax0.set_xlabel("")
        ax[0].set_ylabel("Distance to target")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.2),
                   fancybox=False, shadow=False, ncol=4)
        #plt.show()
        plt.savefig(self.dir + "dofL.png", dpi=300)

    def plot_noise_param(self, programs):
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.3, wspace=0)
        labels = ["Depolarizing", "Amplitude damping"]
        i = 0
        for program in programs:
            noise_param_vs_d1 = []
            noise_param_vs_d2 = []
            noise_param_vs_d3 = []
            noise_param_vs_d4 = []
            path = self.dir + program + "/"
            for file in os.listdir(path):
                with open(path + file, "r") as f:
                    noise_param = float(file[3:])
                    _, t1, t2, t3, d1, d2, d3, d4 = f.read().split("\t")
                    d1 = [float(item) for item in d1[1:len(d1)-2].split(", ")][0]
                    d2 = [float(item) for item in d2[1:len(d2)-2].split(", ")][0]
                    d3 = [float(item) for item in d3[1:len(d3)-2].split(", ")][0]
                    d4 = [float(item) for item in d4[1:len(d4)-2].split(", ")][0]
                    noise_param_vs_d1.append([noise_param, d1])
                    noise_param_vs_d2.append([noise_param, d2])
                    noise_param_vs_d3.append([noise_param, d3])
                    noise_param_vs_d4.append([noise_param, d4])
            noise_param_vs_d1.sort(key=lambda x: x[0])
            noise_param_vs_d2.sort(key=lambda x: x[0])
            noise_param_vs_d3.sort(key=lambda x: x[0])
            noise_param_vs_d4.sort(key=lambda x: x[0])

            noise = np.array(noise_param_vs_d1)[:, :1]
            d1 = np.array(noise_param_vs_d1)[:, 1:].flatten()
            d2 = np.array(noise_param_vs_d2)[:, 1:].flatten()
            d3 = np.array(noise_param_vs_d3)[:, 1:].flatten()
            d4 = np.array(noise_param_vs_d4)[:, 1:].flatten()
            ax[i].plot(noise, d1, label="Mixing")
            ax[i].plot(noise, d2, label="Optimized")
            ax[i].plot(noise, d3, label="Conic")
            ax[i].plot(noise, d4, label="Deterministic")
            print(f"d1: {d1}\nd2: {d2}\nd3: {d3}\nd4: {d4}")

        #           ax[i].plot(rn, d1, label="Mixing")
#           ax[i].plot(rn, d2, label="Optimized")
#           ax[i].plot(rn, d3, label="Conic")
#           ax[i].plot(rn, d4, label="Deterministic")
#           ax[i].set_xlabel("Depth of compilation")

            ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
            ax[i].set_xlabel("Noise parameter")
            ax[i].set_xlim([0.0, 0.04])
            ax[i].set_ylim([0.0, 0.35])
            i += 1
        ax[0].set_ylabel("Distance to target")
        ax[0].set_xticks(np.arange(0.0, 0.04, 0.01))
        #plt.xticks(np.arange(0.0, 0.1, 0.02))
        plt.legend(loc='upper center', bbox_to_anchor=(0, -0.2),
                   fancybox=False, shadow=False, ncol=4)
        plt.show()

        #plt.savefig(self.dir + "dofeta.png", dpi=300)


if __name__ == "__main__":
    res = Results()
    #res.plot_depth(["no-noise-13052022", "depolarizing-13052022", "amplitude-damping-13052022"])
    res.plot_noise_param(["depolarizing-eta-13052022", "amplitude-damping-eta-31052022"])#, "amplitude-damping-13052022"])
