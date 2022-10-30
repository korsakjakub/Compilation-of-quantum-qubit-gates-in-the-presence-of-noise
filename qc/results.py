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
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.3, wspace=0)
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
            ax[i].plot(rn, pd[0], label="Mixing")
            ax[i].plot(rn, pd[1], label="Optimized")
            ax[i].plot(rn, pd[2], label="Conic")
            ax[i].plot(rn, pd[3], label="Deterministic")
            ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
            ax[i].set_xlabel("Depth of compilation")
            i += 1

        plt.xticks([3, 7, 12])
        ax[0].set_ylabel("Distance to target")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.2),
                   fancybox=False, shadow=False, ncol=4)
        plt.savefig(self.dir + "dofL.png", dpi=300)

    def plot_noise_param(self, programs):
        fig, ax = plt.subplots(nrows=1, ncols=len(programs), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.3, wspace=0)
        labels = ["Depolarizing", "Amplitude damping"]
        i = 0
        for program in programs:
            path = self.dir + program + "/"
            dvseta = {}
            noise_params = []
            for file in os.listdir(path):
                with open(path + file, "r") as f:
                    noise_params.append(file.split("V", 1)[1])
                    d = [np.loadtxt(path + file)[:, i] for i in range(4)]
                    d = [np.mean(col) for col in d]
                    dvseta[float(noise_params[-1])] = d
            pd = np.array([dvseta[i] for i in sorted(dvseta)]).T
            noise_params = sorted(noise_params)


            ax[i].plot(noise_params, pd[0], label="Mixing")
            ax[i].plot(noise_params, pd[1], label="Optimized")
            ax[i].plot(noise_params, pd[2], label="Conic")
            ax[i].plot(noise_params, pd[3], label="Deterministic")

            ax[i].annotate(labels[i], xy=(0.1, 0.9), xycoords="axes fraction")
            ax[i].set_xlabel("Noise parameter")
            #ax[i].set_xlim([0.0, 0.1])
            #ax[i].set_ylim([0.0, 0.5])
            i += 1
        ax[0].set_ylabel("Distance to target")
        ax[0].set_xticks(np.arange(0.0, 0.1, 0.01))
        #plt.xticks(np.arange(0.0, 0.1, 0.02))
        plt.legend(loc='upper center', bbox_to_anchor=(0, -0.2),
                   fancybox=False, shadow=False, ncol=4)
        #plt.show()
        plt.savefig(self.dir + "dofeta.png", dpi=300)


if __name__ == "__main__":
    res = Results()
    depth = ["no-noise-28072022", "depolarizing-28072022", "amplitude-damping-26072022"]
    noise = ["depolarizing-eta-29072022", "amplitude-damping-eta-26072022"]
    #res.plot_depth(["no-noise-13052022", "depolarizing-13052022", "amplitude-damping-13052022"])
    #  res.plot_noise_param(["depolarizing-eta-13052022", "amplitude-damping-eta-31052022"])#, "amplitude-damping-13052022"])
    #res.plot_depth(depth)
    res.plot_noise_param(noise)

