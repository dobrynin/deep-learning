class WordEncodingDictionary:
    def __init__(self):
        self.word_to_code_dict = {}
        self.code_to_word_dict = {}

    def word_to_code(self, word):
        if word not in self.word_to_code_dict:
            code = len(self.word_to_code_dict)
            self.word_to_code_dict[word] = code
            self.code_to_word_dict[code] = word

        return self.word_to_code_dict[word]

    def code_to_word(self, code):
        if code not in self.code_to_word_dict:
            raise f"Code {code} not recorded!"

        return self.code_to_word_dict[code]

    def encode(self, text):
        codes = set()
        for word in text.split():
            codes.add(self.word_to_code(word))
        return codes

    def decode(self, codes):
        cls = type(codes)
        return cls(map(self.code_to_word, codes))
