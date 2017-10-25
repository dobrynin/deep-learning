import IPython.display
from lib.newtons_method_solver import NewtonsMethodSolver
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class NewtonsMethodAnimationBuilder:
    MARKER_SIZE = 15
    SLEEP = 2000

    def __init__(self, f, df, x_range, num_steps):
        self.f = f
        self.df = df
        self.x_range = x_range
        self.num_steps = num_steps

    def plot_f(self):
        y = self.f(self.x_range)
        self.axes.plot(self.x_range, y, "-")
        self.axes.set_ylim(bottom = min(y), top = max(y))

        # Draw coordinate axes
        self.axes.spines['left'].set_position('zero')
        self.axes.spines['bottom'].set_position('zero')
        self.axes.spines['right'].set_color('none')
        self.axes.spines['top'].set_color('none')

    def plot_tangent_line(self):
        x_i = self.solver.x_i
        y_i = self.f(x_i)
        slope = self.df(x_i)

        # Plot tangent function
        y = slope * (self.x_range - x_i) + y_i
        self.axes.plot(self.x_range, y, "-")
        self.axes.plot(x_i, y_i, ".", ms = type(self).MARKER_SIZE)

    def paint_frame(self, frame_idx):
        self.axes.clear()

        x_i, y_i = self.solver.x_i, self.solver.y_i
        self.axes.set_title(
            f"Step #{frame_idx}: (x_i, y_i) = ({x_i:0.2f}, {y_i:0.2f})"
        )
        self.plot_f()
        self.plot_tangent_line()

    def run_step(self, frame_idx):
        self.paint_frame(frame_idx)
        self.solver.step()

    def run(self, x_0):
        self.solver = NewtonsMethodSolver(
            f = self.f, df = self.df, x_0 = x_0
        )

        # Open a figure to draw on
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)

        animation = matplotlib.animation.FuncAnimation(
            fig = self.figure,
            func = self.run_step,
            frames = self.num_steps,
            interval = type(self).SLEEP,
            init_func = lambda: None
        )

        # We don't want to display the figure itself. We're just using
        # it for generating the frames of the animation
        plt.close(self.figure)

        return animation

class NewtonsMethodExample:
    NUM_STEPS = 5
    X_RANGE = np.arange(0, +4, 0.01)
    X_START = 0.5

    @classmethod
    def f(cls, x):
        return 8 *(x ** 2) - 20
    @classmethod
    def df(cls, x):
        return 16 * x

    @classmethod
    def run(cls):
        animation_builder = NewtonsMethodAnimationBuilder(
            f = cls.f,
            df = cls.df,
            x_range = cls.X_RANGE,
            num_steps = cls.NUM_STEPS
        )

        animation = animation_builder.run(x_0 = cls.X_START)
        IPython.display.display(
            IPython.display.HTML(animation.to_html5_video())
        )
