from timeit import default_timer as timer

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from tqdm import tqdm

from qc import results, linear_programs, channel

if __name__ == "__main__":
    for _ in range(16):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
        res = results.Results()
        start = timer()
        amount = 10

        program = linear_programs.Program(min_length=1, max_length=13,
                                          wg=WordGenerator(gates, length=0, cascader=Cascader()))
        targets = linear_programs.generate_target(amount)

        program.targets = targets
        program.noise_type = channel.Noise.AmplitudeDamping
        for vv in tqdm(range(10)):
            vis = round(0.01 + 1e-2 * vv, 2)
            ch = channel.Channel(vis=vis, noise=program.noise_type)
            r = program.distribute_calculations_channels(channel=ch, threads=amount)
            res.write(r, vis,
                      "amplitude-damping-22072022")  # "depolarizing-eta-13052022")##"depolarizing-13052022")
        end = timer()
        print(f'czas: {end - start} s')
