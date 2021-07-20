# Generate all possible words of given length


class WordGenerator:

    def __init__(self, input_set: list, length: int):
        self.output = []
        self.input_set = input_set
        self.length = length

    def generate_words(self) -> list:
        self.__generate_words_rec("", self.length)
        self._remove_unnecessary(self.length)
        return self.output

    def __generate_words_rec(self, word: str, length: int) -> None:
        if length == 0:
            self.output.append(word)
            return
        for i in range(len(self.input_set)):
            self.__generate_words_rec(word + self.input_set[i], length - 1)

    def generate_words_shorter_than(self) -> list:
        for i in range(self.length):
            self.__generate_words_rec("", i + 1)
            self._remove_unnecessary(i + 1)
        return self.output

    def _remove_unnecessary(self, length: int = 0) -> None:
        if length == 1:
            return
        if length == 0:
            length = self.length
        for letter in self.input_set:
            self.output.remove(letter * length)
