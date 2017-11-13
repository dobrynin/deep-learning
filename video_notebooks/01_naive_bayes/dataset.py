import numpy as np
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

    def split(self, ratio):
        training_dataset = Dataset(
            self.word_encoding_dictionary,
            [],
            []
        )
        test_dataset = Dataset(
            self.word_encoding_dictionary,
            [],
            []
        )

        for ham_email in self.ham_emails:
            if np.random.uniform() < ratio:
                training_dataset.ham_emails.append(ham_email)
            else:
                test_dataset.ham_emails.append(ham_email)
        for spam_email in self.spam_emails:
            if np.random.uniform() < ratio:
                training_dataset.spam_emails.append(spam_email)
            else:
                test_dataset.spam_emails.append(spam_email)

        return (training_dataset, test_dataset)
