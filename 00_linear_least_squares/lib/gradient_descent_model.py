import numpy as np
from .multivariate_model import MultivariateModel

class GradientDescentModel(MultivariateModel):
    def __init__(self, theta0, theta1, learning_rate):
        super().__init__(theta0, theta1)
        self.learning_rate = learning_rate

    def gradient(self, dataset):
        first_error_deriv_wrt_theta0 = np.sum(
            2 * (self(dataset.x) - dataset.y) * 1
        )

        first_error_deriv_wrt_theta1 = np.sum(
            2 * (self(dataset.x) - dataset.y) * dataset.x
        )

        # I'm dividing by the number of examples so that the
        # calculation is insensitive to training set size.
        return (
            first_error_deriv_wrt_theta0 / dataset.x.shape[0],
            first_error_deriv_wrt_theta1 / dataset.x.shape[0]
        )

    def improve(self, dataset):
        deriv_wrt_theta0, deriv_wrt_theta1 = self.gradient(dataset)
        self.theta0 -= self.learning_rate * deriv_wrt_theta0
        self.theta1 -= self.learning_rate * deriv_wrt_theta1
        return (self.theta0, self.theta1)
