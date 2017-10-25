from .dataset import Dataset
from .normalized_dataset import NORMALIZED_DATASET
import numpy as np

class SkewedDataset(Dataset):
    def __init__(self):
        self.x, self.y = (
            NORMALIZED_DATASET.x + 1,
            NORMALIZED_DATASET.y
        )

SKEWED_DATASET = SkewedDataset()
