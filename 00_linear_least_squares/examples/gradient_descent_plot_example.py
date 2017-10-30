from lib.gradient_descent_model import GradientDescentModel
from lib.multivariate_optimization_plot_animation import MultivariateOptimizationPlotAnimation
from lib.normalized_dataset import NORMALIZED_DATASET

class GradientDescentPlotExample:
    LEARNING_PARAMS = [
        (0.1, 40),
    ]

    @classmethod
    def run_optimization(cls, learning_rate, num_steps):
        model = GradientDescentModel(
            theta0 = 0.0,
            theta1 = 0.0,
            learning_rate = learning_rate,
        )
        animation = MultivariateOptimizationPlotAnimation(
            NORMALIZED_DATASET,
            model,
            num_steps = num_steps,
            sleep = 100,
        )
        animation.run()
        print(
            f"theta0: {animation.unnormalized_theta0():0.2f} | "
            f"theta1: {animation.unnormalized_theta1():0.2f}"
        )

    @classmethod
    def run(cls):
        for (learning_rate, num_steps) in cls.LEARNING_PARAMS:
            print(f"Learning_rate = {learning_rate}")
            cls.run_optimization(
                learning_rate = learning_rate,
                num_steps = num_steps,
            )
