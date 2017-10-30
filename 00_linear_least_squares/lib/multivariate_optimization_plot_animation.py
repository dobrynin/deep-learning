import IPython.display
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class MultivariateOptimizationPlotAnimation:
    FIG_SIZE = (5, 5)
    SLEEP = 1000

    def __init__(
            self,
            dataset,
            model,
            num_steps,
            sleep = SLEEP,
    ):
        self.dataset = dataset
        self.model = model
        self.theta0s, self.theta1s = (
            [self.model.theta0], [self.model.theta1]
        )
        self.num_steps = num_steps
        self.sleep = sleep

    def unnormalized_theta0(self):
        return (
            self.dataset.y_stddev * self.model.theta0
            - (
                self.dataset.y_stddev
                / self.dataset.x_stddev
            ) * self.model.theta1 * self.dataset.x_mean
            + self.dataset.y_mean
        )

    def unnormalized_theta1(self):
        return (
            self.dataset.y_stddev
            / self.dataset.x_stddev
        ) * self.model.theta1

    def frame(self, axes, frame_idx):
        self.model.improve(self.dataset)

        axes.clear()
        axes.plot(
            self.dataset.unnormalized_x,
            self.dataset.unnormalized_y,
            '.'
        )

        x_min, x_max = (
            np.min(self.dataset.unnormalized_x),
            np.max(self.dataset.unnormalized_x)
        )
        x_range = np.arange(x_min, x_max, (x_max - x_min) / 100)
        y_values = (
            self.unnormalized_theta0()
            + self.unnormalized_theta1() * x_range
        )

        axes.plot(x_range, y_values)

        axes.set_title(
            f"Step #{frame_idx} | "
            f"theta0: {self.unnormalized_theta0():0.2f} | "
            f"theta1: {self.unnormalized_theta1():0.2f}"
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
