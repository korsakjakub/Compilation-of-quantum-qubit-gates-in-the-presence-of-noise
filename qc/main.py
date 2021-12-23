from timeit import default_timer as timer

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from tqdm import tqdm

import qc

if __name__ == "__main__":
    for _ in range(1):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z', 'I']
        results = qc.results.Results()
        start = timer()
        program_name = "channels"
        amount = 1

        program = qc.linear_programs.Program(min_length=1, max_length=7, wg=WordGenerator(gates, length=0,
                                                                                          cascader=Cascader()))
        targets = qc.linear_programs.generate_target(program_name, amount)

        program.targets = targets
        program.noise_type = qc.channel.Noise.Depolarizing
        for vv in tqdm(range(1)):
            vis = round(1.00 - 1e-4 * vv, 4)
            channel = qc.channel.Channel(vis=vis, noise=qc.channel.Noise.PauliY)
            res = program.threaded_program(channel=channel,
                                           program=program_name,
                                           threads=amount)
            results.write(res, vis, program_name)
        end = timer()
        print(f'czas: {end - start} s')
