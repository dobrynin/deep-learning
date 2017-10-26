import IPython.display
from lib.multivariate_plot import MultivariatePlot
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class MultivariateOptimizationAnimation:
    GRANULARITY = 50
    THETA0_RANGE = np.arange(-2, 2, 1 / GRANULARITY)
    THETA1_RANGE = np.arange(-1, 3, 1 / GRANULARITY)

    EPSILON = 0.40
    FIG_SIZE = (5, 5)
    HEAD_WIDTH = 0.075
    SLEEP = 1000

    def __init__(
            self,
            dataset,
            model,
            num_steps,
            sleep = SLEEP):
        self.dataset = dataset
        self.model = model
        self.theta0s, self.theta1s = (
            [self.model.theta0], [self.model.theta1]
        )
        self.num_steps = num_steps
        self.sleep = sleep
        self.did_plot_contours = False

    def draw_arrow(self, axes, p1, p2, color, linestyle = 'solid'):
        axes.arrow(
            x = p1[0],
            y = p1[1],
            dx = p2[0] - p1[0],
            dy = p2[1] - p1[1],
            color = color,
            head_width = type(self).HEAD_WIDTH,
            length_includes_head = True,
            linestyle = linestyle,
        )

    def draw_last_update_calculation(self, axes):
        # Draw no arrows if the point didn't change enough.
        update = np.array([
            np.abs(self.theta0s[-1] - self.theta0s[-2]),
            np.abs(self.theta1s[-1] - self.theta1s[-2])
        ])
        if np.min(update) < type(self).EPSILON:
            return

        for idx, theta1 in enumerate(self.theta1s[-2:]):
            self.draw_arrow(
                axes = axes,
                p1 = np.array([self.theta0s[-2], theta1]),
                p2 = np.array([self.theta0s[-1], theta1]),
                color = 'green',
                linestyle = ['solid', 'dashed'][idx],
            )

        for idx, theta0 in enumerate(self.theta0s[-2:]):
            self.draw_arrow(
                axes = axes,
                p1 = np.array([theta0, self.theta1s[-2]]),
                p2 = np.array([theta0, self.theta1s[-1]]),
                color = 'red',
                linestyle = ['solid', 'dashed'][idx],
            )

    def draw_all_steps(self, axes):
        for i in range(1, len(self.theta0s)):
            self.draw_arrow(
                axes = axes,
                p1 = np.array([self.theta0s[i - 1], self.theta1s[i - 1]]),
                p2 = np.array([self.theta0s[i], self.theta1s[i]]),
                color = "blue"
            )

    def frame(self, axes, frame_idx):
        if not self.did_plot_contours:
            plot = MultivariatePlot(
                self.dataset,
                theta0_range = type(self).THETA0_RANGE,
                theta1_range = type(self).THETA1_RANGE,
            )
            plot.plot_error_contours(axes = axes)
            self.did_plot_contours = True

        self.model.improve(self.dataset)

        self.theta0s.append(self.model.theta0)
        self.theta1s.append(self.model.theta1)

        self.draw_all_steps(axes)
        self.draw_last_update_calculation(axes)

        axes.set_title(
            f"Step #{frame_idx} | "
            f"theta0: {self.model.theta0:0.2f} | "
            f"theta1: {self.model.theta1:0.2f}"
        )

    def run(self):
        figure = plt.figure(figsize = type(self).FIG_SIZE)
        axes = figure.add_subplot(111)

        animation = matplotlib.animation.FuncAnimation(
            fig = figure,
            func = lambda frame_idx: self.frame(axes, frame_idx),
            frames = self.num_steps,
            interval = self.sleep,
            init_func = lambda: None
        )

        plt.close(figure)

        IPython.display.display(
            IPython.display.HTML(animation.to_html5_video())
        )
