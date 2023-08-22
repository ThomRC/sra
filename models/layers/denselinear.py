import typing as tp  # NOQA

import cupy as cp
import chainer.functions as F
from chainer import link, variable, initializers, types, utils

from relu_moments import relu_moments

class DenseLinear(link.Link):
    def __init__(self,
                 in_size: tp.Optional[int],
                 out_size: tp.Optional[int] = None,
                 bias: bool = False,
                 initial_w: tp.Optional[types.InitializerSpec] = None,
                 initial_bias: tp.Optional[types.InitializerSpec] = None,
                 **kwargs):
        super(DenseLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.in_size = in_size
        self.out_size = out_size

        with self.init_scope():

            # Set weights and biases.
            w_initializer = initializers._get_initializer(initial_w)
            self.W = variable.Parameter(w_initializer)
            if bias:
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)
            else:
                self.b = None

    def _initialize_params(self, in_size: int) -> None:
        self.W.initialize((self.out_size, in_size))  # type: ignore
        scale = 1. / cp.sqrt(self.W.shape[1])
        self.W *= scale

        if self.b is not None:
            self.b.initialize

    def _set_config(self, config):
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self):
        return 'in_size={}, out_size={}, bias={}'.format(
            self.in_size, self.out_size, self.b is not None)

    def relu_moment_propagation(self, x_m, x_v, cov_ub = None, w_grad = False, ortho_ws = True):
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)

        if w_grad:
            self.orthonormalize()
            W = self.ortho_w
        else:
            W = self.ortho_w.array

        mean_s = F.linear(x_m,W)
        if ortho_ws:
            var_s = F.linear(x_v,W**2)
        else:
            aux = self.xp.zeros((x_m.shape[1],x_m.shape[1],x_m.shape[1]))
            aux = self.xp.fill_diagonal(aux,1)
            cov_h = x_v @ aux
            var_s = F.linear(x_v,W**2)

        h_m, h_v = relu_moments(mean_s, var_s)

        return mean_s, var_s, h_m, h_v