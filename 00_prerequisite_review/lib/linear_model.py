import numpy as np

class LinearModel:
    def __init__(self, theta0, theta1):
        self.theta0 = theta0
        self.theta1 = theta1

    def plot(self, axes, x_values, *args, **kwargs):
        yhat_values = self(x_values)
        axes.plot(x_values, yhat_values, *args, **kwargs)

    def __call__(self, x_values):
        return self.theta0 + self.theta1 * x_values

    def plot_errors(self, axes, dataset):
        yhat_values = self(dataset.x)
        axes.vlines(dataset.x, dataset.y, yhat_values, color = "r")

    def average_squared_error(self, dataset):
        yhat_values = self(dataset.x)
        return np.sum((yhat_values - dataset.y) ** 2) / len(dataset.x)
