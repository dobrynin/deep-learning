from lib.multivariate_optimization_animation import MultivariateOptimizationAnimation
from lib.naive_differentiable_mulivariate_model import NaiveDifferentiableMultivariateModel, AlternatingDifferentiableMultivariateModel
from lib.skewed_dataset import SKEWED_DATASET

class NaiveSkewedOptimizationExample:
    @classmethod
    def run(cls):
        model = NaiveDifferentiableMultivariateModel(
            theta0 = 1.5, theta1 = 0.0
        )
        animation = MultivariateOptimizationAnimation(
            SKEWED_DATASET,
            model,
            num_steps = 8,
        )
        animation.run()

class AlternatingMultivariateOptimizationExample:
    @classmethod
    def run(cls):
        model = AlternatingDifferentiableMultivariateModel(
            theta0 = 1.5, theta1 = 0.0
        )
        animation = MultivariateOptimizationAnimation(
            SKEWED_DATASET,
            model,
            num_steps = 8,
        )
        animation.run()
