import os.path
import pickle

DATA_DIR = os.path.join(
    os.getcwd(),
    "data/"
)

class Dataset:
    def __init__(
            self, word_encoding_dictionary, ham_emails, spam_emails
    ):
        self.word_encoding_dictionary = word_encoding_dictionary
        self.ham_emails = ham_emails
        self.spam_emails = spam_emails

    INSTANCE = None
    @classmethod
    def get(cls):
        if not cls.INSTANCE:
            with open(os.path.join(DATA_DIR, 'data.p'), 'rb') as f:
                cls.INSTANCE = pickle.load(f)
        return cls.INSTANCE
