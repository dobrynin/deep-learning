class Result:
    def __init__(self, score_cutoff, num_spams_identified, recall):
        self.score_cutoff, self.num_spams_identified, self.recall = (
            score_cutoff, num_spams_identified, recall
        )

class NaiveBayesModel:
    def __init__(self, fps, use_negative_features):
        self.fps = fps
        self.use_negative_features = use_negative_features

    def score_email(self, email):
        spam_score = 1.0
        for code in self.fps.code_counts:
            if code in email.codes:
                spam_score *= (
                    self.fps.code_given_class_prob(code).to_odds().spam_odds
                )
            elif self.use_negative_features:
                spam_score *= (
                    self.fps.no_code_given_class_prob(code).to_odds().spam_odds
                )

        return spam_score

    def recall_for_false_positive_rate(self, dataset, limit):
        ham_scores = list(map(
            lambda email: self.score_email(email),
            dataset.ham_emails
        ))
        ham_scores.sort(key = lambda score: -score)

        spam_scores = list(map(
            lambda email: self.score_email(email),
            dataset.spam_emails
        ))

        score_cutoff = ham_scores[int(len(ham_scores) * limit)]
        num_spams_identified = sum(
            [1 if s > score_cutoff else 0 for s in spam_scores]
        )
        recall = (
            num_spams_identified / len(dataset.spam_emails)
        )

        return Result(
            score_cutoff = score_cutoff,
            num_spams_identified = num_spams_identified,
            recall = recall,
        )
