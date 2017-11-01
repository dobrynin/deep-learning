class FeatureProbabilitiesExplorer:
    @classmethod
    def best_spam_features(
        cls, fps, limit = 20, present_features = True
    ):
        return cls.best_features(
            fps, limit, -1, present_features = present_features
        )

    @classmethod
    def best_ham_features(
        cls, fps, limit = 20, present_features = True
    ):
        return cls.best_features(
            fps, limit, +1, present_features = present_features
        )

    @classmethod
    def best_features(cls, fps, limit, multiplier, present_features):
        if present_features:
            prob_ratio_fn = fps.code_prob_ratio
            code_counts = lambda code: fps.code_counts[code]
        else:
            prob_ratio_fn = fps.no_code_prob_ratio
            code_counts = fps.no_code_counts

        codes = list(fps.code_counts.keys())
        code_prob_ratios = [{
            'code': code,
            'reach': code_counts(code),
            'feature_probability_ratio': (
                prob_ratio_fn(code).feature_probability_ratio
            )
        } for code in codes]
        code_prob_ratios.sort(key = lambda code_prob_ratio: (
            multiplier * code_prob_ratio['feature_probability_ratio']
        ))
        return code_prob_ratios[:limit]

    @classmethod
    def print_features_list(
        cls, features_list, word_encoding_dictionary
    ):
        for code_prob_ratio in features_list:
            code, reach, feature_probability_ratio = (
                code_prob_ratio['code'],
                code_prob_ratio['reach'],
                code_prob_ratio['feature_probability_ratio']
            )
            word = word_encoding_dictionary.code_to_word(code)
            print(
                f"{code} | {word} | reach: {reach} | "
                f"feature_probability_ratio: {feature_probability_ratio:0.2f}:1"
            )
