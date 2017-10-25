from lib.normalized_dataset import NORMALIZED_DATASET
from lib.multivariate_plot import MultivariatePlot
import numpy as np

class MultivariatePlotExample:
    GRANULARITY = 50
    THETA0_RANGE = np.arange(-2, 2, 1 / GRANULARITY)
    THETA1_RANGE = np.arange(-1, 3, 1 / GRANULARITY)

    @classmethod
    def run(cls):
        MultivariatePlot.plot(
            dataset = NORMALIZED_DATASET,
            theta0_range = cls.THETA0_RANGE,
            theta1_range = cls.THETA1_RANGE,
        )
