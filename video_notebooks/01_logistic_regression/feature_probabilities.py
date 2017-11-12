import numpy as np

# Keeps track of how many time a word occurs in ham or spam emails.
class Counts:
    def __init__(self, ham_count = 0, spam_count = 0):
        self.ham_count, self.spam_count = (
            ham_count, spam_count
        )

    def total_count(self):
        return self.ham_count + self.spam_count

    def __repr__(self):
        return self.__dict__.__repr__()

# Keeps a map of codes to feature probability ratio.
class FeatureProbabilities:
    def __init__(self):
        self.class_counts = Counts()
        self.code_counts = {}

    @classmethod
    def from_dataset(cls, dataset):
        return cls.from_emails(
            ham_emails = dataset.ham_emails,
            spam_emails = dataset.spam_emails
        )

    @classmethod
    def from_emails(cls, ham_emails, spam_emails):
        fps = cls()

        for ham_email in ham_emails:
            fps.add_email(ham_email, True)
        for spam_email in spam_emails:
            fps.add_email(spam_email, False)

        return fps

    def add_email(self, email, is_ham_email):
        if is_ham_email:
            self.class_counts.ham_count += 1
        else:
            self.class_counts.spam_count += 1

        for code in email.codes:
            if code not in self.code_counts:
                self.code_counts[code] = Counts()

            if is_ham_email:
                self.code_counts[code].ham_count += 1
            else:
                self.code_counts[code].spam_count += 1

    def code_prob_ratio(self, code):
        _code_counts = self.code_counts[code]

        code_given_spam_prob = (
            _code_counts.spam_count / self.class_counts.spam_count
        )

        if _code_counts.ham_count == 0:
            return np.inf

        code_given_ham_prob = (
            _code_counts.ham_count / self.class_counts.ham_count
        )

        return code_given_spam_prob / code_given_ham_prob

    def filter(self, limit):
        fps = type(self)()

        fps.class_counts = self.class_counts

        for code in self.code_counts:
            if self.code_counts[code].total_count() < limit: continue
            fps.code_counts[code] = self.code_counts[code]

        return fps
