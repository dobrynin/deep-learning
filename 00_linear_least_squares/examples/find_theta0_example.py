import IPython.display
from lib.dataset import DATASET
from lib.differentiable_linear_model import DifferentiableLinearModel
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class FindTheta0Animation:
    NUM_STEPS = 4
    SLEEP = 2000

    @staticmethod
    def frame(axes, dataset, generator_model, step_idx):
        axes.clear()

        x_values = np.arange(0, 100, 1.0)
        average_squared_error = generator_model.average_squared_error(
            dataset
        )

        dataset.plot(axes)
        generator_model.plot(axes, x_values, "g-")
        axes.set_title(
            f"Step #{step_idx} | "
            f"theta0: {generator_model.theta1:0.2f} | "
            f"Avg SSE: {average_squared_error:0.2f}"
        )

        generator_model.improve_theta1(dataset)

    @classmethod
    def run(cls, dataset = DATASET, fixed_theta0 = 100):
        figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)

        generator_model = DifferentiableLinearModel(
            theta0 = fixed_theta0, theta1 = 0.0
        )

        frame_fn = lambda step_idx: (
            cls.frame(axes, dataset, generator_model, step_idx)
        )

        animation = matplotlib.animation.FuncAnimation(
            figure,
            frame_fn,
            frames = cls.NUM_STEPS,
            interval = cls.SLEEP,
            init_func = lambda: None
        )

        plt.close(figure)

        IPython.display.display(
            IPython.display.HTML(animation.to_html5_video())
        )
