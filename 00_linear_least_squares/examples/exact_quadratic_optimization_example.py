from lib.skewed_dataset import SKEWED_DATASET
from lib.exact_quadratic_optimizer import ExactQuadraticOptimizer

class ExactQuadraticOptimizationExample:
    @classmethod
    def run(cls):
        optimizer = ExactQuadraticOptimizer(SKEWED_DATASET)

        theta0, theta1 = 1.5, 0
        initial_gradient = optimizer.gradient(theta0, theta1)
        print(
            f"Initial | "
            f"theta0 = {theta0:.2f} | "
            f"theta1 = {theta1:.2f} | "
            f"gradient = {initial_gradient}"
        )

        updated_theta = optimizer.calculate_delta(theta0, theta1)
        updated_gradient = optimizer.gradient(
            updated_theta[0], updated_theta[1]
        )

        print(
            f"Calculated | "
            f"theta0 = {updated_theta[0]:.2f} | "
            f"theta1 = {updated_theta[1]:.2f} | "
            f"gradient = {updated_gradient}"
        )
