class NaiveBayesModel:
    def __init__(self, fps):
        self.fps = fps

    def score_email(self, email):
        odds_email_is_spam = (
            self.fps.class_counts.spam_count
            / self.fps.class_counts.ham_count
        )

        for code in self.fps.code_counts:
            if code not in email.codes:
                odds_email_is_spam *= self.fps.no_code_prob_ratio(code)
            else:
                odds_email_is_spam *= self.fps.code_prob_ratio(code)

        return odds_email_is_spam

    def score_emails(self, emails):
        return list(
            map(self.score_email, emails)
        )
