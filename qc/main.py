from timeit import default_timer as timer

from tqdm import tqdm

from qc.channel import *
from qc.linear_programs import Program
from qc.results import Results

if __name__ == "__main__":
    for _ in range(1):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z', 'I']
        channel = Channel(vis=vis, noise=Noise.PauliY)
        writer = Results()
        start = timer()
        program_name = "channels"
        amount = 1

        program = Program(min_length=1, max_length=5)
        targets = program.generate_target(program_name, amount)

        program.targets = targets
        program.noise_type = noise_type
        for vv in tqdm(range(1)):
            vis = round(1.00 - 1e-4 * vv, 4)
            res = program.threaded_program(gates=gates, channel=channel,
                                           program=program_name,
                                           threads=amount)
            writer.write_results(res, vis, program_name)
        end = timer()
        print(f'czas: {end - start} s')
