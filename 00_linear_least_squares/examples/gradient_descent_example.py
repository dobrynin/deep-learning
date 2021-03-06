from lib.gradient_descent_model import GradientDescentModel
from lib.multivariate_optimization_animation import MultivariateOptimizationAnimation
from lib.skewed_dataset import SKEWED_DATASET

class GradientDescentExample:
    LEARNING_PARAMS = [
        (0.01, 40),
        (0.3, 10),
        (0.39, 10),
    ]

    @classmethod
    def run_optimization(cls, learning_rate, num_steps):
        model = GradientDescentModel(
            theta0 = 1.0,
            theta1 = 2.0,
            learning_rate = learning_rate,
        )
        animation = MultivariateOptimizationAnimation(
            SKEWED_DATASET,
            model,
            num_steps = num_steps,
            head_width = 0.0,
            sleep = 100,
            draw_last_update_calculation = False
        )
        animation.run()

    @classmethod
    def run(cls):
        for (learning_rate, num_steps) in cls.LEARNING_PARAMS:
            print(f"Learning_rate = {learning_rate}")
            cls.run_optimization(
                learning_rate = learning_rate,
                num_steps = num_steps,
            )

if __name__ == '__main__':
    GradientDescentExample.run()
