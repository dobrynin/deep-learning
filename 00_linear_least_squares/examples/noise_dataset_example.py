from .find_theta0_example import FindTheta0Animation
from lib.dataset import Dataset, DATASET
from lib.differentiable_linear_model import DifferentiableLinearModel

class NoiseDataSet(Dataset):
    def __init__(self, true_dataset):
        self.x, self.y = true_dataset.x, true_dataset.noises

class NoiseDatasetAnimation:
    def run(dataset = DATASET):
        return FindTheta0Animation.run(
            NoiseDataSet(dataset), fixed_theta0 = 0.0
        )
