from dataset import Dataset
from feature_probabilities import FeatureProbabilities, Counts
from feature_probabilities_explorer import FeatureProbabilitiesExplorer

d = Dataset.get()
fps = FeatureProbabilities.from_dataset(d).filter(limit = 100)

limited_code = d.word_encoding_dictionary.word_to_code("limited")
offer_code = d.word_encoding_dictionary.word_to_code("offer")

limited_count = fps.code_counts[limited_code]
offer_count = fps.code_counts[offer_code]

both_count = Counts()
for e in d.ham_emails:
    if limited_code not in e.codes: continue
    if offer_code not in e.codes: continue

    both_count.ham_count += 1
for e in d.spam_emails:
    if limited_code not in e.codes: continue
    if offer_code not in e.codes: continue

    both_count.spam_count += 1

print(
    f"limited: {limited_count}"
)
print(
    f"offer: {offer_count}"
)
print(
    f"both: {both_count}"
)
