import IPython.display
from lib.newtons_method_optimization_solver import NewtonsMethodOptimizationSolver
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class NewtonsMethodOptimizationAnimationBuilder:
    MARKER_SIZE = 15
    SLEEP = 2000

    def __init__(self, f, df, d2f, x_range):
        self.f, self.df, self.d2f = f, df, d2f
        self.x_range = x_range

    def plot_f_and_df(self):
        y = self.f(self.x_range)
        dy = self.df(self.x_range)

        self.axes.plot(self.x_range, y, "-", label = "f")
        self.axes.plot(self.x_range, dy, "-", label = "df")

        # Set limits and give a little headroom
        self.axes.set_ylim(
            bottom = np.min(np.concatenate((y, dy))) - 1,
            top = np.max(np.concatenate((y, dy))) + 1
        )

        # Plot axes
        self.axes.spines['left'].set_position('zero')
        self.axes.spines['bottom'].set_position('zero')
        self.axes.spines['right'].set_color('none')
        self.axes.spines['top'].set_color('none')

    def plot_tangent_line(self):
        # Plot tangent
        tangent_y = self.solver.tangent_line(self.x_range)
        self.axes.plot(self.x_range, tangent_y, "-", label = "tangent")

        x_i, y_i = (
            self.solver.x_i, self.solver.y_i
        )
        self.axes.plot(x_i, y_i, ".", ms = type(self).MARKER_SIZE)

    def paint_frame(self, frame_idx):
        self.axes.clear()

        x_i, y_i, dy_i = (
            self.solver.x_i, self.solver.y_i, self.solver.dy_i
        )

        self.axes.set_title(
            f"Step #{frame_idx}: "
            f"(x_i, df(x_i), f(x_i)) "
            f"= ({x_i:0.2f}, {dy_i:0.2f}, {y_i:0.2f})"
        )
        
        self.plot_f_and_df()
        self.plot_tangent_line()
        self.axes.legend()
        
    def run_step(self, frame_idx):
        self.paint_frame(frame_idx)
        self.solver.step()

    def run(self, num_steps, x_0):
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)
        self.solver = NewtonsMethodOptimizationSolver(
            f = self.f, df = self.df, d2f = self.d2f, x_0 = x_0
        )

        animation = matplotlib.animation.FuncAnimation(
            fig = self.figure,
            func = self.run_step,
            frames = num_steps,
            interval = type(self).SLEEP,
            init_func = lambda: None
        )

        plt.close(self.figure)

        return animation

class NewtonsMethodOptimizationExample:
    NUM_STEPS = 5
    X_RANGE = np.arange(-9, +6, 0.01)
    X_START = 6

    @classmethod
    def f(cls, x):
        return (x - 3) * (x + 4) * (x + 5)
    @classmethod
    def df(cls, x):
        return (x + 4) * (x + 5) + (x - 3) * (x + 5) + (x - 3) * (x + 4)
    @classmethod
    def d2f(cls, x):
        return (x + 4) + (x + 5) + (x - 3) + (x + 5) + (x - 3) + (x + 4)

    @classmethod
    def run(cls, x_0 = None):
        if x_0 is None: x_0 = cls.X_START
        builder = NewtonsMethodOptimizationAnimationBuilder(
            f = cls.f,
            df = cls.df,
            d2f = cls.d2f,
            x_range = cls.X_RANGE
        )

        animation = builder.run(num_steps = cls.NUM_STEPS, x_0 = x_0)
        IPython.display.display(
            IPython.display.HTML(animation.to_html5_video())
        )
