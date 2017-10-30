from .dataset import Dataset, DATASET
import numpy as np

class NormalizedDataset(Dataset):
    def __init__(self, dataset):
        self.x_mean = np.mean(dataset.x)
        self.x_stddev = np.sqrt(np.var(dataset.x))
        self.x = (dataset.x - self.x_mean) / self.x_stddev
        self.unnormalized_x = dataset.x

        self.y_mean = np.mean(dataset.y)
        self.y_stddev = np.sqrt(np.var(dataset.y))
        self.y = (dataset.y - self.y_mean) / self.y_stddev
        self.unnormalized_y = dataset.y

NORMALIZED_DATASET = NormalizedDataset(DATASET)
