import autograd.numpy as np
import torch

from reg.nn.npy.nn_ag import NNRegressor as agNetwork
from reg.nn.torch.nn_torch import NNRegressor as torchNetwork
from reg.nn.npy.nn_npy import NNRegressor as npNetwork

to_float = lambda arr: torch.from_numpy(arr).float()


if __name__ == '__main__':

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # prepare data
    set = load_boston()

    x, y = set['data'], set['target']
    y = y[:, np.newaxis]

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.1)

    nb_in = x.shape[-1]
    nb_out = y.shape[-1]

    # fit neural network with numpy
    nn_np = npNetwork([nb_in, 5, 5, nb_out], nonlin='relu', output='linear', loss='mse')

    print('using numpy network:')
    nn_np.fit(yt, xt, nb_epochs=2500, batch_size=64, lr=1e-3)

    print('numpy', 'train:', 'cost=', nn_np.cost(yt, xt)[0])
    print('numpy', 'test:', 'cost=', nn_np.cost(yv, xv)[0])

    # fit neural network with autograd
    nn_ag = agNetwork([nb_in, 5, 5, nb_out], nonlin='relu', output='linear', loss='mse')

    print('using autograd network:')
    nn_ag.fit(yt, xt, nb_epochs=2500, batch_size=64, lr=1e-3)

    print('autograd', 'train:', 'cost=', nn_ag.cost(yt, xt))
    print('autograd', 'test:', 'cost=', nn_ag.cost(yv, xv))

    # fit neural network with pytorch
    nn_torch = torchNetwork([nb_in, 5, 5, nb_out], nonlin='relu')

    print('using pytorch network:')
    nn_torch.fit(yt, xt, nb_epochs=2500, batch_size=64, lr=1e-3)

    yt = to_float(yt).reshape(-1, 1)
    yv = to_float(yv).reshape(-1, 1)

    _yt = to_float(nn_torch.predict(xt)).reshape(-1, 1)
    _yv = to_float(nn_torch.predict(xv)).reshape(-1, 1)

    print('pytorch', 'train:', 'cost=', nn_torch.criterion(yt, _yt))
    print('pytorch', 'test:', 'cost=', nn_torch.criterion(yv, _yv))
