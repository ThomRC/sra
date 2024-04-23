import typing as tp  # NOQA

import cupy as cp
import chainer.functions as F
from chainer import link, variable, initializers, types, utils

from models.activations.relu_moments import relu_moments

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

    def forward(self, x):
        raise NotImplementedError

    def relu_moment_propagation(self, x_m, x_v, w_grad = False, layer_num = None, x_var = None):
        """ Computes the pre-activation's mean and variance vectors and calls the relu_moments function

        Args:
            x_m: previous layer's mean vector
            x_v: previous layer's variance vector (we assume independent activations, i.e., zero covariances
            w_grad: boolean telling whether the gradients are needed or not

        Returns:

        """

        if w_grad:
            self.orthonormalize() # carries orthonormalization only in the case of needing the W gradients
            W = self.ortho_w
        else:
            W = self.ortho_w.array

        mean_s = F.linear(x_m,W)
        
        if layer_num == 0:
            var_s = variable.Variable(cp.zeros_like(mean_s.array)) + x_var
        else:
            var_s = F.linear(x_v,W**2)

        h_m, h_v = relu_moments(mean_s, var_s)

        return mean_s, var_s, h_m, h_v