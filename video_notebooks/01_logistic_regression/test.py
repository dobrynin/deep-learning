from dataset import Dataset
from feature_probabilities import FeatureProbabilities
from feature_probabilities_explorer import FeatureProbabilitiesExplorer

d = Dataset.get()
fps = FeatureProbabilities.from_dataset(d).filter(limit = 100)
best_spam_features = (
    FeatureProbabilitiesExplorer.best_spam_features(fps)
)

FeatureProbabilitiesExplorer.print_features_list(
    best_spam_features,
    d.word_encoding_dictionary,
)
