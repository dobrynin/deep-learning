class NewtonsMethodSolver:
    def __init__(self, f, df, x_0):
        self.f = f
        self.df = df
        self.x_i = x_0
        self.y_i = f(x_0)

    def step(self):
        self.x_i = self.x_i - self.f(self.x_i) / self.df(self.x_i)
        self.y_i = self.f(self.x_i)
        return self.x_i
