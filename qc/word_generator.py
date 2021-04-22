# Generate all possible words of given length
class WordGenerator:

    def __init__(self, input_set, length):
        self.output: list = []
        self.input_set = input_set
        self.length = length

    def generate_words(self) -> list:
        self.generate_words_rec("", self.length)
        return self.output

    def generate_words_rec(self, word, length) -> None:
        if length == 0:
            self.output.append(word)
            return
        for i in range(len(self.input_set)):
            self.generate_words_rec(word + self.input_set[i], length-1)
