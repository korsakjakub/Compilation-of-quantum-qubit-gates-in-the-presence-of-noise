from channel import Noise


class Config:
    DIR = "files/"
    WORDS_DIR = DIR + "words/"
    OUTPUTS_DIR = DIR + "outputs/"
    FIGURES_DIR = DIR + "figures/"
    THREADS = 10
    MAIN_ITERATIONS = 8
    NOISE_ITERATIONS = 10
    INITIAL_NOISE_PARAMETER = 0.01
    NOISE_PARAMETER_ADDITION = 1e-2
    MIN_LENGTH = 12
    MAX_LENGTH = 12
    OUTPUT_PATH = "amplitude-damping-eta-26072022"
    NOISE_TYPE = Noise.AmplitudeDamping
