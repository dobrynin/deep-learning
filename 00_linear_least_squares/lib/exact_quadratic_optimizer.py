import numpy as np
import numpy.linalg

class ExactQuadraticOptimizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_yhat(self, theta0, theta1):
        return theta0 + theta1 * self.dataset.x

    def partial_wrt_theta0(self, theta0, theta1):
        yhat = self.calculate_yhat(theta0, theta1)
        return np.sum(2 * (yhat - self.dataset.y) * 1)

    def partial_wrt_theta1(self, theta0, theta1):
        yhat = self.calculate_yhat(theta0, theta1)
        return np.sum(2 * (yhat - self.dataset.y) * self.dataset.x)

    def second_partial_wrt_theta0(self):
        return 2 * len(self.dataset.x)

    def second_partial_wrt_theta1(self):
        return 2 * np.dot(self.dataset.x, self.dataset.x)

    def mixed_partial(self):
        return 2 * np.sum(self.dataset.x)

    def gradient(self, theta0, theta1):
        return np.array([
            self.partial_wrt_theta0(theta0, theta1),
            self.partial_wrt_theta1(theta0, theta1)
        ])

    def hessian(self):
        return np.array([
            [
                self.second_partial_wrt_theta0(),
                self.mixed_partial()
            ],
            [
                self.mixed_partial(),
                self.second_partial_wrt_theta1()
            ]
        ])

    def calculate_delta(self, theta0, theta1):
        inverted_hessian = numpy.linalg.inv(self.hessian())
        delta = inverted_hessian.dot(-self.gradient(theta0, theta1))
        return np.array([theta0, theta1]) + delta
