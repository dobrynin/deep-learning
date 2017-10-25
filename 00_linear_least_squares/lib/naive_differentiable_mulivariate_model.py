from .multivariate_model import MultivariateModel
import numpy as np

class NaiveDifferentiableMultivariateModel(MultivariateModel):
    def __init__(self, theta0, theta1):
        super().__init__(theta0, theta1)

    def calculate_improved_theta0(self, dataset):
        first_error_deriv_wrt_theta0 = np.sum(
            2 * (dataset.y - self(dataset.x)) * -np.ones_like(dataset.x)
        )
        second_error_derivative_wrt_theta0 = np.sum(
            2 * -np.ones_like(dataset.x) * -np.ones_like(dataset.x)
        )

        delta = (
            first_error_deriv_wrt_theta0
            / second_error_derivative_wrt_theta0
        )

        return self.theta0 - delta

    def calculate_improved_theta1(self, dataset):
        first_error_deriv_wrt_theta1 = np.sum(
            2 * (dataset.y - self(dataset.x)) * (-dataset.x)
        )
        second_error_derivative_wrt_theta1 = np.sum(
            2 * (-dataset.x) * (-dataset.x)
        )

        delta = (
            first_error_deriv_wrt_theta1
            / second_error_derivative_wrt_theta1
        )

        return self.theta1 - delta

    def improve(self, dataset):
        # Notice how I calculate improved new_theta0, new_theta1
        # before changing either.
        new_theta0 = self.calculate_improved_theta0(dataset)
        new_theta1 = self.calculate_improved_theta1(dataset)
        self.theta0, self.theta1 = new_theta0, new_theta1

        return (self.theta0, self.theta1)
