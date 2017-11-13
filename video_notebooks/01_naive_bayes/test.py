from dataset import Dataset
from feature_probabilities import FeatureProbabilities, Counts
from feature_probabilities_explorer import FeatureProbabilitiesExplorer
from naive_bayes_model import NaiveBayesModel, recall_for_false_positive_rate

d = Dataset.get()
fps = FeatureProbabilities.from_dataset(d).filter(limit = 100)

model = NaiveBayesModel(fps)

ham_scores = model.score_emails(d.ham_emails)
spam_scores = model.score_emails(d.spam_emails)

cutoffs = [10000, 1000, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
for cutoff in cutoffs:
    result = recall_for_false_positive_rate(
        score_cutoff = cutoff,
        ham_scores = ham_scores,
        spam_scores = spam_scores,
    )

    print(cutoff)
    print(result)
