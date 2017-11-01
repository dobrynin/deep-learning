import numpy as np

class Counts:
    def __init__(self, ham_count = 0, spam_count = 0):
        self.ham_count, self.spam_count = (
            ham_count, spam_count
        )

    def total_count(self):
        return self.ham_count + self.spam_count

    def __repr__(self):
        return self.__dict__.__repr__()

class PriorClassProbabilities:
    def __init__(self, class_counts):
        self.ham_prior_prob = (
            class_counts.ham_count / class_counts.total_count()
        )
        self.spam_prior_prob = (
            class_counts.spam_count / class_counts.total_count()
        )

class ConditionalFeatureProbabilityRatio:
    def __init__(self, feature_counts, class_counts):
        self.prob_feature_given_ham = (
            feature_counts.ham_count / class_counts.ham_count
        )
        self.prob_feature_given_spam = (
            feature_counts.spam_count / class_counts.spam_count
        )

        if (self.prob_feature_given_ham != 0):
            self.feature_probability_ratio = (
                self.prob_feature_given_spam
                / self.prob_feature_given_ham
            )
        else:
            self.feature_probability_ratio = np.inf

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

    def add_email(self, email, is_ham_email):
        if is_ham_email:
            self.class_counts.ham_count += 1
        else:
            self.class_counts.spam_count += 1

        for code in email.codes:
            self._check_code_added(code)

            if is_ham_email:
                self.code_counts[code].ham_count += 1
            else:
                self.code_counts[code].spam_count += 1

    def class_prior_probs(self):
        return PriorClassProbabilities(self.class_counts)

    def code_prob_ratio(self, code):
        return ConditionalFeatureProbabilityRatio(
            feature_counts = self.code_counts[code],
            class_counts = self.class_counts
        )

    def filter(self, reach_limit):
        filtered_fps = (type(self))()
        filtered_fps.class_counts = self.class_counts
        for (code, counts) in self.code_counts.items():
            if counts.total_count() < reach_limit: continue
            filtered_fps.code_counts[code] = counts

        return filtered_fps

    def no_code_counts(self, code):
        code_counts = self.code_counts[code]
        return Counts(
            ham_count = (
                self.class_counts.ham_count - code_counts.ham_count
            ),
            spam_count = (
                self.class_counts.spam_count - code_counts.spam_count
            )
        )

    def no_code_prob_ratio(self, code):
        return ConditionalFeatureProbabilityRatio(
            feature_counts = self.no_code_counts(code),
            class_counts = self.class_counts
        )

    def _check_code_added(self, code):
        if code in self.code_counts: return
        self.code_counts[code] = Counts(
            ham_count = 0,
            spam_count = 0,
        )
