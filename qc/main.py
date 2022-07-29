from timeit import default_timer as timer

from qworder.cascading_rules import Cascader
from qworder.word_generator import WordGenerator
from tqdm import tqdm

from results import Results
from linear_programs import Program, generate_target
from channel import Channel, Noise
from config import Config

if __name__ == "__main__":
    for i in range(Config.MAIN_ITERATIONS):
        gates = ['H', 'T', 'R', 'X', 'Y', 'Z']
        res = Results()
        start = timer()
        amount = Config.THREADS

        program = Program(min_length=Config.MIN_LENGTH, max_length=Config.MAX_LENGTH,
                          wg=WordGenerator(gates, length=0, cascader=Cascader()))
        targets = generate_target(amount)

        program.targets = targets
        program.noise_type = Noise.Depolarizing
        for vv in tqdm(range(Config.NOISE_ITERATIONS)):
            vis = round(Config.INITIAL_NOISE_PARAMETER + Config.NOISE_PARAMETER_ADDITION * vv, 2)
            ch = Channel(vis=vis, noise=program.noise_type)
            r = program.distribute_calculations_channels(channel=ch, threads=amount)
            res.write(r, vis, Config.OUTPUT_PATH)
        end = timer()
        print(f'Iteracja {i+1}/{Config.MAIN_ITERATIONS}\tczas: {end - start} s')
