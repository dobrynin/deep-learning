from lib.config import DATA_DIR
import os.path
import pickle
import zlib

class Dataset:
    def __init__(
            self, word_encoding_dictionary, ham_emails, spam_emails
    ):
        self.word_encoding_dictionary = word_encoding_dictionary
        self.ham_emails = ham_emails
        self.spam_emails = spam_emails

    def split(self, ratio):
        datasetA = self._split(ratio, 0)
        datasetB = self._split(ratio, 1)
        return (datasetA, datasetB)

    def _split(self, ratio, mode):
        split_ham_emails, split_spam_emails = [], []
        emails_pairs = [
            (self.ham_emails, split_ham_emails),
            (self.spam_emails, split_spam_emails)
        ]
        for (emails, split_emails) in emails_pairs:
            for email in emails:
                h = zlib.crc32(email.path.encode())
                p = h / (2**32 - 1)
                if (mode == 0 and p < ratio) or (mode == 1 and p >= ratio):
                    split_emails.append(email)

        return Dataset(
            self.word_encoding_dictionary,
            ham_emails = split_ham_emails,
            spam_emails = split_spam_emails
        )

class RawDataset(Dataset):
    INSTANCE = None

    @classmethod
    def get(cls):
        if not cls.INSTANCE:
            with open(os.path.join(DATA_DIR, 'data.p'), 'rb') as f:
                cls.INSTANCE = pickle.load(f)
        return cls.INSTANCE
