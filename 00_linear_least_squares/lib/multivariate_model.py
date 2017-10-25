import numpy as np

class MultivariateModel:
    @classmethod
    def calculate_error(cls, dataset, theta0, theta1):
        yhat = theta0 + theta1 * dataset.x
        return np.sum((dataset.y - yhat) ** 2)

    @classmethod
    def calculate_errors(cls, dataset, theta0s, theta1s):
        errors = np.zeros_like(theta0s)
        for (i, j) in np.ndindex(*theta0s.shape):
            errors[i, j] = cls.calculate_error(
                dataset, theta0s[i, j], theta1s[i, j]
            )
        return errors

    def __init__(self, theta0, theta1):
        self.theta0 = theta0
        self.theta1 = theta1

    def __call__(self, x_values):
        return self.theta0 + self.theta1 * x_values

    def error(self, dataset):
        return MultivariateModel.calculate_error(
            dataset, self.theta0, self.theta1
        )
