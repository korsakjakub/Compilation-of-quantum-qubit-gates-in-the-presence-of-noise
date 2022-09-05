from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt

from config import Config


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
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True, figsize=(7.67, 4.5))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.2, wspace=0)
        i = 0
        labels = ["No noise", "Depolarizing", "Amplitude damping"]
        for program in programs:
            path = self.dir + program + "/"
            dvsL = {}
            for file in os.listdir(path):
                # depth = file[:-4]
                depth = file.split("V", 1)[0]
                d = [np.loadtxt(path + file)[:, i] for i in range(4)]
                d = [np.mean(col) for col in d]
                dvsL[int(depth)] = d
            pd = np.array([dvsL[i] for i in sorted(dvsL)]).T
            rn = np.arange(1, len(pd[0]) + 1, 1)
            ax[i].plot(rn, pd[0], label="$t_1$")
            ax[i].plot(rn, pd[1], label="$t_2$")
            ax[i].plot(rn, pd[2], label="$t_3$")
            ax[i].plot(rn, pd[3], label="Deterministic")
            ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
            ax[i].set_xlabel("Depth of compilation")
            i += 1

        plt.xticks([3, 7, 12])
        ax[0].set_ylabel("Distance to target")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.14),
                   fancybox=False, shadow=False, ncol=4)
        plt.savefig(self.dir + "dofL.png", dpi=300)

    def plot_noise_param(self, programs):
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True, figsize=(7.67, 4.5))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.2, wspace=0)
        labels = ["Depolarizing", "Amplitude damping"]
        i = 0
        for program in programs:
            noise_param_vs_d1 = []
            noise_param_vs_d2 = []
            noise_param_vs_d3 = []
            noise_param_vs_d4 = []
            path = self.dir + program + "/"
            for file in os.listdir(path):
                noise_param = float(file[3:])*100
                d1, d2, d3, d4 = np.genfromtxt(path+file, delimiter="\t").T
                noise_param_vs_d1.append([noise_param, np.mean(d1)])
                noise_param_vs_d2.append([noise_param, np.mean(d2)])
                noise_param_vs_d3.append([noise_param, np.mean(d3)])
                noise_param_vs_d4.append([noise_param, np.mean(d4)])
            noise_param_vs_d1.sort(key=lambda x: x[0])
            noise_param_vs_d2.sort(key=lambda x: x[0])
            noise_param_vs_d3.sort(key=lambda x: x[0])
            noise_param_vs_d4.sort(key=lambda x: x[0])

            noise = np.array(noise_param_vs_d1)[:, :1]
            d1 = np.array(noise_param_vs_d1)[:, 1:].flatten()
            d2 = np.array(noise_param_vs_d2)[:, 1:].flatten()
            d3 = np.array(noise_param_vs_d3)[:, 1:].flatten()
            d4 = np.array(noise_param_vs_d4)[:, 1:].flatten()
            ax[i].plot(noise, d1, label="$t_1$")
            ax[i].plot(noise, d2, label="$t_2$")
            ax[i].plot(noise, d3, label="$t_3$")
            ax[i].plot(noise, d4, label="Deterministic")
            ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
            ax[i].set_xlabel("Noise parameter [%]")
            ax[i].set_xlim([0.0, 0.09])
            ax[i].set_ylim([0.0, 0.5])
            i += 1
        ax[0].set_ylabel("Distance to target")
        ax[0].set_xticks([1, 3, 5, 7, 9])
        plt.legend(loc='upper center', bbox_to_anchor=(0, -0.15),
                   fancybox=False, shadow=False, ncol=4)
        plt.savefig(self.dir + "dofeta.png", dpi=300)


if __name__ == "__main__":
    res = Results()
    output_paths_depth = ["no-noise-28072022", "depolarizing-28072022", "amplitude-damping-26072022"]
    output_paths_noise = ["depolarizing-eta-29072022", "amplitude-damping-eta-26072022"]
    res.plot_noise_param(output_paths_noise)
    # res.plot_depth(output_paths_depth)
