from channel import Noise


class Config:
    DIR = "files/"
    WORDS_DIR = DIR + "words/"
    OUTPUTS_DIR = DIR + "outputs/"
    FIGURES_DIR = DIR + "figures/"
    THREADS = 1
    MAIN_ITERATIONS = 1
    NOISE_ITERATIONS = 1
    INITIAL_NOISE_PARAMETER = 0.00
    NOISE_PARAMETER_ADDITION = 1e-2
    MIN_LENGTH = 1
    MAX_LENGTH = 5
    OUTPUT_PATH = "test"
    NOISE_TYPE = Noise.AmplitudeDamping
