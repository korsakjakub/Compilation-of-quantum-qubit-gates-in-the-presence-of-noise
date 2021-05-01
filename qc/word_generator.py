# Generate all possible words of given length
class WordGenerator:

    def __init__(self, input_set: list, length: int):
        self.output: list = []
        self.input_set = input_set
        self.length = length

    def generate_words(self) -> list:
        self.__generate_words_rec("", self.length)
        return self.output

    def __generate_words_rec(self, word: str, length: int) -> None:
        if length == 0:
            self.output.append(word)
            return
        for i in range(len(self.input_set)):
            self.__generate_words_rec(word + self.input_set[i], length-1)
