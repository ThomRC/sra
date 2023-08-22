import typing as tp  # NOQA

import chainer.functions as F
from chainer import utils
from denselinear import DenseLinear
from bjorck_ortho import bjorck_orthonormalize, get_safe_bjorck_scaling, min_ortho_iter
from updater_adam import Adam

class BjorckLinear(DenseLinear, Adam):
    def __init__(self, in_size, layer_num, config=None, **kwargs):
        DenseLinear.__init__(self, in_size, **kwargs)
        Adam.__init__(self, **kwargs)
        self._set_config(config)
        self.layer_num = layer_num
        self.ortho_w = None
        self.iter = config['iter']
        self.dynamic_iter = config['dynamic_iter']

    def forward(self, x, n_batch_axes: int = 1, train = False):
        # kwargs:  'train', 'dynamic_iter'
        if not hasattr(self, 'iter'):
            self.iter = self.config['iter']

        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)
        # Scale the values of the matrix to make sure the singular values are less than or equal to 1.
        if train and self.iter > 0:
            self.orthonormalize()
        elif train and self.iter == 0:
            self.ortho_w = self.W
        elif self.dynamic_iter:
            if self.last_epoch % 10 == 0 and self.last_epoch > 0:
                self.iter = min_ortho_iter(self.W,
                                           beta=self.config['beta'],
                                           iters=self.iter,
                                           order=self.config['order'])
                print(self.iter)

        return F.linear(x, self.ortho_w, self.b, n_batch_axes=n_batch_axes)

    def orthonormalize(self):
        if self.config['safe_scaling']:
            scaling = get_safe_bjorck_scaling(self.W)
        else:
            scaling = 1.0

        self.ortho_w = bjorck_orthonormalize(self.W.T / scaling,
                                             beta=self.config['beta'],
                                             iters=self.iter,
                                             order=self.config['order']).T
