import numpy as np

class NaiveBayesModel:
    def __init__(self, fps, use_negative_features):
        self.fps = fps
        self.use_negative_features = use_negative_features

    def _base_spam_score(self):
        return (
            self.fps.class_prior_probs().spam_prior_prob
            / self.fps.class_prior_probs().ham_prior_prob
        )

    def _build_feature_weights(self):
        # Note that because of filtering, some entries will remain
        # 1.0. That's fine.
        positive_code_prob_ratios = np.ones(
            max(self.fps.code_counts.keys()) + 1
        )
        negative_code_prob_ratios = np.ones(
            len(positive_code_prob_ratios)
        )

        for code in self.fps.code_counts:
            positive_code_prob_ratios[code] = (
                self.fps.code_prob_ratio(code).feature_probability_ratio
            )
            negative_code_prob_ratios[code] = (
                self.fps.no_code_prob_ratio(code).feature_probability_ratio
            )

        return (positive_code_prob_ratios, negative_code_prob_ratios)

    def score_email(
        self,
        email,
        base_spam_score,
        positive_code_prob_ratios,
        negative_code_prob_ratios
    ):
        spam_score = base_spam_score
        for code in self.fps.code_counts.keys():
            if code in email.codes:
                spam_score *= positive_code_prob_ratios[code]
            elif self.use_negative_features:
                spam_score *= negative_code_prob_ratios[code]

        return spam_score

    def score_emails(self, emails):
        base_spam_score = self._base_spam_score()
        (positive_code_prob_ratios, negative_code_prob_ratios) = (
            self._build_feature_weights()
        )

        return map(
            lambda email: self.score_email(
                email = email,
                base_spam_score = base_spam_score,
                positive_code_prob_ratios = positive_code_prob_ratios,
                negative_code_prob_ratios = negative_code_prob_ratios,
            ),
            emails
        )
