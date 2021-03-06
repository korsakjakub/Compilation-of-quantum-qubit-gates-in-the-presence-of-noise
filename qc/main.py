from timeit import default_timer as timer

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from tqdm import tqdm

import qc
from qc import results, linear_programs

if __name__ == "__main__":
    for _ in range(1):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
        results = qc.results.Results()
        start = timer()
        program_name = "channels"
        amount = 1

        program = qc.linear_programs.Program(min_length=1, max_length=13, wg=WordGenerator(gates, length=0,
                                                                                          cascader=Cascader()))
        targets = qc.linear_programs.generate_target(program_name, amount)

        program.targets = targets
        program.noise_type = qc.channel.Noise.AmplitudeDamping
        for vv in tqdm(range(9)):
            vis = round(0.01 + 1e-2 * vv, 2)
            channel = qc.channel.Channel(vis=vis, noise=program.noise_type)
            res = program.threaded_program(channel=channel,
                                           program=program_name,
                                           threads=amount)
            results.write(res, vis, "amplitude-damping-31052022")#"depolarizing-eta-13052022")##"depolarizing-13052022")
        end = timer()
        print(f'czas: {end - start} s')
