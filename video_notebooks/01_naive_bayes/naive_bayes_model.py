class NaiveBayesModel:
    def __init__(self, fps):
        self.fps = fps

    def score_email(self, email):
        odds_email_is_spam = (
            self.fps.class_counts.spam_count
            / self.fps.class_counts.ham_count
        )

        for code in email.codes:
            if code not in self.fps.code_counts: continue
            odds_email_is_spam *= self.fps.code_prob_ratio(code)

        return odds_email_is_spam

    def score_emails(self, emails):
        return list(
            map(self.score_email, emails)
        )

def recall_for_false_positive_rate(score_cutoff, ham_scores, spam_scores):
    num_false_positives = len(
        list(filter(lambda ham_score: ham_score > score_cutoff, ham_scores))
    )
    num_true_positives = len(
        list(filter(lambda spam_score: spam_score > score_cutoff, spam_scores))
    )

    false_positive_rate = num_false_positives / len(ham_scores)
    recall = num_true_positives / len(spam_scores)

    return {
        'false_positive_rate': false_positive_rate,
        'recall': recall
    }
