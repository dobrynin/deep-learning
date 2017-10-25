from .linear_model import LinearModel
import numpy as np

class DifferentiableLinearModel(LinearModel):
    def __init__(self, theta0, theta1):
        super().__init__(theta0, theta1)

    def improve_theta1(self, dataset):
        first_error_deriv_wrt_theta1 = np.sum(
            2 * (dataset.y - self(dataset.x)) * (-dataset.x)
        )
        second_error_derivative_wrt_theta1 = np.sum(
            2 * (-dataset.x) * (-dataset.x)
        )

        delta = -(
            first_error_deriv_wrt_theta1
            / second_error_derivative_wrt_theta1
        )

        self.theta1 += delta
