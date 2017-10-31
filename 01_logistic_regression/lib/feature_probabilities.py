import math

class Counts:
    def __init__(self, total_count = 0, ham_count = 0, spam_count = 0):
        self.total_count, self.ham_count, self.spam_count = (
            total_count, ham_count, spam_count
        )

    def to_probs(self):
        return Probs(self)

    def to_odds(self):
        return self.to_probs().to_odds()

    def __repr__(self):
        return self.__dict__.__repr__()

class Probs:
    def __init__(self, counts):
        self.ham_prob, self.spam_prob = (
            counts.ham_count / counts.total_count,
            counts.spam_count / counts.total_count
        )

    def to_odds(self):
        return Odds(self)

    def __repr__(self):
        return self.__dict__.__repr__()

class Odds:
    def __init__(self, probs):
        if probs.spam_prob != 0.0:
            self.ham_odds = probs.ham_prob / probs.spam_prob
        else:
            self.ham_odds = math.inf

        if probs.ham_prob != 0.0:
            self.spam_odds = probs.spam_prob / probs.ham_prob
        else:
            self.spam_odds = math.inf

    def __repr__(self):
        return self.__dict__.__repr__()

class FeatureProbabilities:
    def __init__(self):
        self.class_counts = Counts()
        self.code_counts = {}

    @classmethod
    def from_emails(cls, ham_emails, spam_emails):
        fps = cls()

        for ham_email in ham_emails:
            fps.add_email(ham_email, True)
        for spam_email in spam_emails:
            fps.add_email(spam_email, False)

        return fps

    @classmethod
    def filter(cls, fps, limit):
        filtered_fps = cls()
        filtered_fps.class_counts = fps.class_counts
        for (code, counts) in fps.code_counts.items():
            if counts.total_count < limit: continue
            filtered_fps.code_counts[code] = counts

        return filtered_fps

    def add_email(self, email, is_ham_email):
        for code in email.codes:
            self.check_code_added(code)

            self.class_counts.total_count += 1
            self.code_counts[code].total_count += 1
            if is_ham_email:
                self.class_counts.ham_count += 1
                self.code_counts[code].ham_count += 1
            else:
                self.class_counts.spam_count += 1
                self.code_counts[code].spam_count += 1

    def class_probs(self):
        return self.class_counts.to_probs()

    def code_given_class_prob(self, code):
        return self.code_counts[code].to_probs()

    def check_code_added(self, code):
        if code in self.code_counts: return
        self.code_counts[code] = Counts(
            total_count = 0,
            ham_count = 0,
            spam_count = 0,
        )
