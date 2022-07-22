from timeit import default_timer as timer

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from tqdm import tqdm

from qc import results, linear_programs, channel

if __name__ == "__main__":
    for _ in range(1):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
        results = results.Results()
        start = timer()
        amount = 1

        program = linear_programs.Program(min_length=1, max_length=13, wg=WordGenerator(gates, length=0,
                                                                                          cascader=Cascader()))
        targets = linear_programs.generate_target(amount)

        program.targets = targets
        program.noise_type = channel.Noise.AmplitudeDamping
        for vv in tqdm(range(9)):
            vis = round(0.01 + 1e-2 * vv, 2)
            channel = channel.Channel(vis=vis, noise=program.noise_type)
            res = program.threaded_program(channel=channel,
                                           threads=amount)
            results.write(res, vis, "amplitude-damping-31052022")#"depolarizing-eta-13052022")##"depolarizing-13052022")
        end = timer()
        print(f'czas: {end - start} s')
