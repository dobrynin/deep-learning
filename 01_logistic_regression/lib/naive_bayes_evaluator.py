import numpy as np

class RecallResult:
    def __init__(self, score_cutoff, num_spams_identified, recall):
        self.score_cutoff, self.num_spams_identified, self.recall = (
            score_cutoff, num_spams_identified, recall
        )

class NaiveBayesEvaluator:
    @classmethod
    def recall_for_false_positive_rates(cls, model, dataset, limits):
        ham_scores = list(model.score_emails(dataset.ham_emails))
        ham_scores.sort(key = lambda score: -score)
        spam_scores = list(model.score_emails(dataset.spam_emails))

        def calculate_result(limit):
            score_cutoff = ham_scores[int(len(ham_scores) * limit)]
            num_spams_identified = sum(
                [1 if s > score_cutoff else 0 for s in spam_scores]
            )
            recall = (
                num_spams_identified / len(dataset.spam_emails)
            )

            return RecallResult(
                score_cutoff = score_cutoff,
                num_spams_identified = num_spams_identified,
                recall = recall,
            )

        return [
            (limit, calculate_result(limit)) for limit in limits
        ]

    @classmethod
    def spam_probability_at_different_scores(cls, model, dataset):
        ham_scores = list(model.score_emails(dataset.ham_emails))
        spam_scores = list(model.score_emails(dataset.spam_emails))

        def calculate_result(limit):
            score_cutoff = limit / (1 - limit)
            print(score_cutoff)
            num_true_positives = sum(
                [1 if s > score_cutoff else 0 for s in spam_scores]
            )
            num_false_positives = sum(
                [1 if s > score_cutoff else 0 for s in ham_scores]
            )
            num_positives = num_true_positives + num_false_positives
            if num_positives == 0:
                return None
            num_expected_positives = sum(
                [(s / (s + 1)) if s > score_cutoff else 0 for s in spam_scores]
            )

            return {
                "limit": limit,
                "expected_rate": num_expected_positives / num_positives,
                "true_rate": num_true_positives / num_positives
            }

        limits = np.arange(0.5, 1.0, 0.1)
        return [
            (limit, calculate_result(limit)) for limit in limits
        ]