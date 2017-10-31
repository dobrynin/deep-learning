from lib.config import DATA_DIR
import os.path
import pickle

class RawDataset:
    INSTANCE = None

    def __init__(
            self, word_encoding_dictionary, ham_emails, spam_emails
    ):
        self.word_encoding_dictionary = word_encoding_dictionary
        self.ham_emails = ham_emails
        self.spam_emails = spam_emails

    @classmethod
    def get(cls):
        if not cls.INSTANCE:
            with open(os.path.join(DATA_DIR, 'data.p'), 'rb') as f:
                cls.INSTANCE = pickle.load(f)
        return cls.INSTANCE
