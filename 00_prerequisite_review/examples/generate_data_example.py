import matplotlib.pyplot as plt

def run(dataset):
    figure = plt.figure()
    axes = figure.add_subplot(111)

    # To check things are working right.
    print(f"x shape: {dataset.x.shape}")
    print(f"y shape: {dataset.y.shape}")

    dataset.plot(axes)
