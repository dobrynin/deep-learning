from lib.linear_model import LinearModel
import matplotlib.pyplot as plt
import numpy as np

def run(dataset):
    x_values = np.arange(0, 100, 1.0)

    generator_model = LinearModel(
        type(dataset).THETA0,
        type(dataset).THETA1
    )

    figure = plt.figure()
    axes = figure.add_subplot(111)
    dataset.plot(axes)
    generator_model.plot(axes, x_values, "g-")
    generator_model.plot_errors(axes, dataset)

    average_sse = generator_model.average_squared_error(dataset)
    print(f"Average Squared Error: {average_sse:0.2f}")
