import IPython.display
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

class CircularMotionAnimation:
    FIG_SIZE = (5, 5)
    FIG_RANGE = 1.25
    NUM_STEPS = 16
    SLEEP = 1000

    def __init__(self):
        self.theta = 0

    def draw_arrow(self, axes, p1, p2, color, linestyle = None):
        axes.arrow(
            x = p1[0],
            y = p1[1],
            dx = p2[0] - p1[0],
            dy = p2[1] - p1[1],
            color = color,
            length_includes_head = True,
            linestyle = linestyle,
            head_width = 0.05,
        )

    def draw_arrows(self, axes):
        self.draw_fixed_vector(axes)
        self.draw_rotating_vector(axes)
        if self.should_draw_projection():
            self.draw_projection(axes)

    def draw_circle(self, axes):
        circle = plt.Circle((0, 0), 1, color = 'blue', fill = False)
        axes.add_artist(circle)

    def draw_fixed_vector(self, axes):
        self.draw_arrow(axes, (0, 0), (0, 1), color = 'green')

    def draw_projection(self, axes):
        projection_endpoint = self.projection_endpoint()
        self.draw_arrow(
            axes,
            (0, 0),
            projection_endpoint,
            color = 'yellow',
        )
        self.draw_arrow(
            axes,
            (0, 1),
            projection_endpoint,
            color = 'green',
            linestyle = 'dotted',
        )

    def draw_rotating_vector(self, axes):
        endpoint = self.rotating_vector_endpoint()
        self.draw_arrow(axes, (0, 0), endpoint, color = 'blue')

        velocity = 0.5 * np.array([
            -np.sin(self.theta), np.cos(self.theta)
        ])
        self.draw_arrow(
            axes,
            endpoint,
            endpoint + velocity,
            color = 'red'
        )

    def frame(self, axes, frame_idx):
        axes.clear()

        fig_range = type(self).FIG_RANGE
        axes.set_xlim(left = -fig_range, right = +fig_range)
        axes.set_ylim(bottom = -fig_range, top = +fig_range)

        self.draw_circle(axes)
        self.draw_arrows(axes)

        self.theta += 2 * np.pi / type(self).NUM_STEPS

    def projection_endpoint(self):
        projection = np.dot(
            np.array([0, 1]),
            self.rotating_vector_endpoint()
        )
        projection_endpoint = (
            projection * self.rotating_vector_endpoint()
        )

        return projection_endpoint

    def rotating_vector_endpoint(self):
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    def should_draw_projection(self):
        projection_endpoint = self.projection_endpoint()
        projection_norm = np.sqrt(
            projection_endpoint.dot(projection_endpoint)
        )
        return (
            (not np.isclose(projection_norm, 0.0))
            and
            (not np.isclose(projection_norm, 1.0))
        )

    def run(self):
        figure = plt.figure(figsize = type(self).FIG_SIZE)
        axes = figure.add_subplot(111)

        animation = matplotlib.animation.FuncAnimation(
            fig = figure,
            func = lambda frame_idx: self.frame(axes, frame_idx),
            frames = type(self).NUM_STEPS,
            interval = type(self).SLEEP,
            init_func = lambda: None
        )

        plt.close(figure)

        IPython.display.display(
            IPython.display.HTML(animation.to_html5_video())
        )
