from __future__ import annotations

from qc.config import Config


class Results(object):

    def __init__(self):
        self.dir = Config.OUTPUTS_DIR
        self.fig_dir = Config.FIGURES_DIR

    def write(self, results: list, vis: float, program: str) -> None:
        for t in results:
            for r in t:
                output_file = open(self.dir + program + "/" + str(r[0]) + "V" + str(vis), "a")
                r = [str(r[i]) for i in range(len(r))]
                output_file.write('\t'.join(r) + '\n')
                output_file.close()


if __name__ == "__main__":
    pass
