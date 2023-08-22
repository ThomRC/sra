import cupy as cp
import cupyx.scipy.special.erf as erf

from chainer import function_node
from chainer.utils import type_check

class ReluMean(function_node.FunctionNode):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('pre_m', 'pre_v'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        eps = 10**-14 # numerical stability constant
        self.retain_inputs((0, 1))
        mean_s, var_s = inputs
        alpha = mean_s/(cp.sqrt(2*var_s) + eps)
        alpha_erf = erf(alpha)
        term1_mean = cp.sqrt(0.5*var_s/cp.pi) * cp.exp(-alpha**2)
        term2_mean = mean_s * (1 + alpha_erf) * 0.5
        return term1_mean + term2_mean,

    def backward(self, indexes, grad_outputs):
        eps = 10**-14 # numerical stability constant
        gy, = grad_outputs
        mean_s, var_s = self.get_retained_inputs()
        mean_s = mean_s.array
        var_s = var_s.array

        alpha = mean_s/(cp.sqrt(2*var_s) + eps)
        alpha_erf = erf(alpha)
        gmean_s = gy * (1 + alpha_erf) * 0.5
        gvar_s = gy * cp.exp(-alpha**2)/(cp.sqrt(8*cp.pi*var_s) + eps)

        return gmean_s, gvar_s

class ReluVar(function_node.FunctionNode):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('pre_m', 'pre_v', 'h_m'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        eps = 10**-14 # numerical stability constant
        self.retain_inputs((0, 1, 2))
        mean_s, var_s, mean_h = inputs

        alpha = mean_s/(cp.sqrt(2 * var_s) + eps)
        alpha_erf = erf(alpha)
        h_v = (var_s/2) * (1 + alpha_erf) - \
              mean_h * (mean_h - mean_s)# + eps
        h_v = cp.clip(h_v, a_min = 0)
        return h_v,

    def backward(self, indexes, grad_outputs):
        eps = 10**-14 # numerical stability constant
        gy, = grad_outputs
        mean_s, var_s, mean_h = self.get_retained_inputs()
        mean_s = mean_s.array
        var_s = var_s.array
        mean_h = mean_h.array
        alpha = mean_s/(cp.sqrt(2*var_s) + eps)
        alpha_erf = erf(alpha)
        gmean_s = gy * (1 - alpha_erf) * mean_h
        gvar_s = gy * ((1 - (mean_s/(cp.sqrt(2*cp.pi*var_s) + eps)) * cp.exp(-alpha**2)) *
                       (1 + alpha_erf) - cp.exp(-2*alpha**2)/cp.pi) * 0.5

        return gmean_s, gvar_s, None


def relu_mean(pre_m, pre_v):
    """ Mean of Rectified Normal distribution (a.k.a., mean of Rectified Linear
        with Gaussian pre-activation) function
    The function computes the mean of the rectified Normal distribution with mean
    ``pre_m`` and variance ``pre_v``.
    Args ``pre_m`` and ``pre_v`` must have the same dimensions (diagonal var.
    matrix) OR ``pre_v`` has one more dimension than ``pre_m`` (var. matrix with
    non-diagonal elements).
    Args:
        pre_m (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        pre_v (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.
    """
    return ReluMean().apply((pre_m, pre_v))[0]


def relu_var(pre_m, pre_v, h_m):
    """ Variance of Rectified Normal distribution (a.k.a., variance of Rectified
        Linear with Gaussian pre-activation) function
    The function computes the variance of the rectified Normal distribution with mean
    ``pre_m`` and variance ``pre_v``.
    Args ``pre_m`` and ``pre_v`` must have the same dimensions (diagonal var.
    matrix) OR ``pre_v`` has one more dimension than ``pre_m`` (var. matrix with
    non-diagonal elements).
    Args:
        pre_m (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        pre_v (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.
    """
    return ReluVar().apply((pre_m, pre_v, h_m))[0]

def relu_moments(pre_m, pre_v):
    h_m = relu_mean(pre_m, pre_v)
    h_v = relu_var(pre_m, pre_v, h_m)

    return h_m, h_v
