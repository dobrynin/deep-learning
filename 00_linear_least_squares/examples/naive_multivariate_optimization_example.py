from lib.multivariate_optimization_animation import MultivariateOptimizationAnimation
from lib.naive_differentiable_mulivariate_model import NaiveDifferentiableMultivariateModel
from lib.normalized_dataset import NORMALIZED_DATASET

class NaiveMultivariateOptimizationExample:
    @classmethod
    def run(cls):
        model = NaiveDifferentiableMultivariateModel(
            theta0 = 1.0, theta1 = 0.0
        )
        animation = MultivariateOptimizationAnimation(
            NORMALIZED_DATASET,
            model,
            num_steps = 1,
        )
        animation.run()
