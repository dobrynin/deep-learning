import numpy as np
# So we generate the same data each time!
np.random.seed(0)

class Dataset:
    NOISE_STDDEV = 25
    NUM_SAMPLES = 100
    THETA0 = 100.0
    THETA1 = 5.0

    @classmethod
    def generate_data(cls):
        xs = []
        ys = []
        noises = []
        for _ in range(cls.NUM_SAMPLES):
            # Sample a random x value between 0 and 100.
            x_observation = np.random.uniform(low = 0.0, high = 100.0)
            y_observation = cls.THETA0 + (x_observation * cls.THETA1)
            # Add some "noise" to y
            noise = np.random.normal(scale = cls.NOISE_STDDEV)
            y_observation += noise

            xs.append(x_observation)
            ys.append(y_observation)
            noises.append(noise)

        # Return numpy arrays of the data values.
        return np.array(xs), np.array(ys), np.array(noises)

    def __init__(self):
        self.x, self.y, self.noises = type(self).generate_data()

    def plot(self, axes):
        axes.plot(self.x, self.y, ".")
        axes.set_ylabel("y")
        axes.set_xlabel("x")
        # Even if all data is positive, show y axis
        axes.set_ylim(ymin = np.min((np.min(self.y), 0)))

        return axes

DATASET = Dataset()
