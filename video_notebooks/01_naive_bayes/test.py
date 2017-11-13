from dataset import Dataset
from feature_probabilities import FeatureProbabilities, Counts
from feature_probabilities_explorer import FeatureProbabilitiesExplorer
from naive_bayes_model import NaiveBayesModel
from recall import recall_for_false_positive_rates

d = Dataset.get()
training_dataset, test_dataset = d.split(0.80)
fps = FeatureProbabilities.from_dataset(training_dataset).filter(limit = 100)

model = NaiveBayesModel(fps)

ham_scores = model.score_emails(test_dataset.ham_emails)
spam_scores = model.score_emails(test_dataset.spam_emails)

recall_results = recall_for_false_positive_rates(
    ham_scores,
    spam_scores,
    limits = [
        0.001,
        0.01,
        0.02,
        0.04,
        0.08
    ]
)

for result in recall_results: print(result)
