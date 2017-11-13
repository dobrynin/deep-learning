# Helper class (see below)
class RecallResult:
    def __init__(self, score_cutoff, num_spams_identified, recall):
        self.score_cutoff, self.num_spams_identified, self.recall = (
            score_cutoff, num_spams_identified, recall
        )

    def __repr__(self):
        return self.__dict__.__repr__()

# Determines what percentage of spam emails are detected if we can
# tolerate a given false positive rate.
# Does this for multiple false positive rate limits.
def recall_for_false_positive_rates(ham_scores, spam_scores, limits):
    ham_scores.sort(key = lambda score: -score)

    def calculate_result(limit):
        score_cutoff = ham_scores[int(len(ham_scores) * limit)]
        num_spams_identified = sum(
            [1 if s > score_cutoff else 0 for s in spam_scores]
        )
        recall = (
            num_spams_identified / len(spam_scores)
        )

        return RecallResult(
            score_cutoff = score_cutoff,
            num_spams_identified = num_spams_identified,
            recall = recall,
        )

    return [
        (limit, calculate_result(limit)) for limit in limits
    ]
