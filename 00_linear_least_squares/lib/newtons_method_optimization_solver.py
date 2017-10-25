class NewtonsMethodOptimizationSolver:
    def __init__(self, f, df, d2f, x_0):
        self.f, self.df, self.d2f = f, df, d2f
        self.x_i = x_0
        self.recalculate()

    def recalculate(self):
        self.y_i = self.f(self.x_i)
        self.dy_i = self.df(self.x_i)
        self.d2y_i = self.d2f(self.x_i)

    def step(self):
        self.x_i = self.x_i - (self.dy_i / self.d2y_i)
        self.recalculate()

    def tangent_line(self, x_range):
        return self.d2y_i * (x_range - self.x_i) + self.dy_i
