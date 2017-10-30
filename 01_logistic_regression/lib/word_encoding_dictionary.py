class WordEncodingDictionary:
    def __init__(self):
        self.word_to_int_dict = {}
        self.int_to_word_dict = {}

    def word_to_int(self, word):
        if word not in self.word_to_int_dict:
            encoding = len(self.word_to_int_dict)
            self.word_to_int_dict[word] = encoding
            self.int_to_word_dict[encoding] = word

        return self.word_to_int_dict[word]

    def int_to_word(self, encoding):
        if encoding not in self.int_to_word_dict:
            raise f"Encoding {encoding} not recorded!"

        return self.int_to_word_dict[encoding]

    def encode(self, content):
        encoding = set()
        for word in content.split():
            encoding.add(self.word_to_int(word))
        return encoding
