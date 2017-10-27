from lib.gradient_descent_model import GradientDescentModel
from lib.multivariate_optimization_animation import MultivariateOptimizationAnimation
from lib.skewed_dataset import SKEWED_DATASET

class GradientDescentExample:
    @classmethod
    def run(cls):
        model = GradientDescentModel(
            theta0 = 1.0,
            theta1 = 2.0,
            learning_rate = 0.05,
        )
        animation = MultivariateOptimizationAnimation(
            SKEWED_DATASET,
            model,
            num_steps = 40,
            head_width = 0.0,
            sleep = 100,
        )
        animation.run()

if __name__ == '__main__':
    GradientDescentExample.run()
