import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .multivariate_model import MultivariateModel
import numpy as np

class MultivariatePlot:
    FIG_SIZE = (5, 5)

    def __init__(self, dataset, theta0_range, theta1_range):
        self.dataset = dataset
        self.theta0s, self.theta1s = np.meshgrid(
            theta0_range, theta1_range
        )

    def plot_error_surface(self, axes = None):
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(111, projection = "3d")

        errors = MultivariateModel.calculate_errors(
            self.dataset, self.theta0s, self.theta1s
        )

        axes.plot_surface(self.theta0s, self.theta1s, errors)
        axes.view_init(azim = -90)

    def plot_error_contours(self, axes = None):
        if axes is None:
            figure = plt.figure(figsize = type(self).FIG_SIZE)
            axes = figure.add_subplot(111)

        errors = MultivariateModel.calculate_errors(
            self.dataset, self.theta0s, self.theta1s
        )

        contour_levels = self.contour_levels(errors)
        contours = axes.contour(
            self.theta0s, self.theta1s, errors, levels = contour_levels
        )
        axes.clabel(contours, fmt = "%02.1g")
        axes.set_xlabel("theta0")
        axes.set_ylabel("theta1")

    def contour_levels(self, errors):
        max_contour = np.max(errors)
        return max_contour / (3.0 ** np.arange(5.0, 0.0, -1.0))

    @classmethod
    def plot(cls, dataset, theta0_range, theta1_range):
        plot = cls(
            dataset = dataset,
            theta0_range = theta0_range,
            theta1_range = theta1_range,
        )

        plot.plot_error_surface()
        plot.plot_error_contours()
