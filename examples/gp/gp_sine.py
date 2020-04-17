import numpy as np
import matplotlib.pyplot as plt

from reg.gp import GPRegressor


if __name__ == "__main__":

    np.random.seed(1337)

    # data
    input = np.linspace(0, 1, 50)
    target = np.sin(input * (2 * np.pi)) + np.random.randn(input.shape[0]) * 0.1

    # build model
    model = GPRegressor(input_size=1)
    model.fit(target, input, preprocess=True)

    output = model.predict(input)

    plt.scatter(input, target, s=5, color='r', zorder=10)
    plt.plot(input, output, '-o', color='b', zorder=1)
    plt.show()
