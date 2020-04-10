import math
import torch

from reg.gp import GPRegressor


if __name__ == "__main__":

    # data
    input = torch.linspace(0, 1, 100)
    target = torch.sin(input * (2 * math.pi)) + torch.randn(input.size()) * 0.1

    # build model
    model = GPRegressor()
    model.fit(target, input)
