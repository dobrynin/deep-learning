class Email:
    def __init__(self, path, content, word_encoding_dictionary, label):
        self.path = path
        self.encoding = word_encoding_dictionary.encode(content)
        self.label = label
