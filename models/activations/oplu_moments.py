import warnings
warnings.filterwarnings("ignore")

import cupy as cp
import cupyx.scipy.special.erf as erf

from chainer import function_node
from chainer.utils import type_check

class OpluMean(function_node.FunctionNode):

    """Orthogonal Permutation Linear Unit mean class"""

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

        delta_m = mat3['{}'.format(units)].T @ mean_s
        sum_v = mat4['{}'.format(units)].T @ var_s

        a = delta_m / cp.sqrt(2 * sum_v)
        p = (1 - erf(a)) / 2
        b = cp.sqrt(sum_v / (2 * cp.pi)) * cp.exp(-a ** 2)

        return mat2['{}'.format(units)] @ mean_s + mat3['{}'.format(units)] @ (p * delta_m - b),

    def backward(self, indexes, grad_outputs):
        eps = 10**-14 # numerical stability constant
        gm, = grad_outputs
        mean_s, var_s = self.get_retained_inputs()
        mean_s = mean_s.array
        var_s = var_s.array

        delta_m = mat3['{}'.format(units)].T @ mean_s
        sum_v = mat4['{}'.format(units)].T @ var_s

        a = delta_m / (cp.sqrt(2 * sum_v) + eps)
        p = (1 - erf(a)) / 2
        c = - 0.5 * (1 / (cp.sqrt(2 * cp.pi * sum_v) + eps)) * cp.exp(-a ** 2)

        g1 = mat4['{}'.format(units)] @ p
        aux_mat = (mat1['{}'.format(units)] - mat2['{}'.format(units)]) @ gm
        gmean_s = mat2['{}'.format(units)] @ gm + aux_mat * g1
        gvar_s = aux_mat * (mat3['{}'.format(units)] @ c)

        return gmean_s, gvar_s

class OpluVar(function_node.FunctionNode):

    """Orthogonal Permutation  Linear Unit variance class"""

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
        self.retain_inputs((0, 1))
        mean_s, var_s = inputs

        delta_m = mat3['{}'.format(units)].T @ mean_s
        delta_v = mat3['{}'.format(units)].T @ var_s
        sum_v = mat4['{}'.format(units)].T @ var_s

        a = delta_m / (cp.sqrt(2 * sum_v) + eps)
        p = (1 - erf(a)) * 0.5

        b = (p * delta_m - cp.sqrt(sum_v / (2 * cp.pi)) * cp.exp(-a ** 2))
        c = mat4['{}'.format(units)] @ ((b - delta_m) * b)

        return mat2['{}'.format(units)] @ var_s + mat3['{}'.format(units)] @ (p * delta_v) - c, c

    def backward(self, indexes, grad_outputs):
        eps = 10**-14 # numerical stability constant
        gv, gcv = grad_outputs
        mean_s, var_s = self.get_retained_inputs()
        mean_s = mean_s.array
        var_s = var_s.array

        delta_m = mat3['{}'.format(units)].T @ mean_s
        delta_v = mat3['{}'.format(units)].T @ var_s
        sqrt_sum_v = mat4['{}'.format(units)].T @ var_s

        norm_m = cp.sqrt(0.5) * delta_m / (sqrt_sum_v + eps)
        p = (1 - erf(norm_m)) * 0.5
        alpha = - (1 / cp.sqrt(2 * cp.pi)) * cp.exp(-norm_m ** 2) # dP1/dmu1
        I = p * delta_m + sqrt_sum_v * alpha
        a = delta_v * alpha / (sqrt_sum_v + eps)
        b = p * (2 * I - delta_m) - I
        c = alpha * (alpha + norm_m * (p - 0.5))
        d = 0.5 * alpha / (sqrt_sum_v + eps) + p
        g1 = mat1['{}'.format(units)] @ gv
        g2 = mat2['{}'.format(units)] @ gv

        gmean_s = (g2 - g1) * (mat3['{}'.format(units)] @ a) - (g2 + g1 - 2 * gcv) * (mat3['{}'.format(units)] @ b)
        gvar_s = (g1 + g2 - 2 * gc) * (mat4['{}'.format(units)] @ c) + (g1 - g2) * (mat3['{}'.format(units)] @ d + mat4['{}'.format(units)] @ p) + g2

        return gmean_s, gvar_s

def oplu_mean(pre_m, pre_v):
    """ Mean of Orthogonal Permutation Normal distribution (a.k.a., mean of Rectified Linear with Gaussian pre-activation) function
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
            A variable holding an array representing the mean of a layer with OPLU units and independent normally distributed pre-activations
    """
    return OpluMean().apply((pre_m, pre_v))[0]

def oplu_var(pre_m, pre_v, h_m):
    """ Variance of Orthogonal Permutation Normal distribution (a.k.a., variance of Rectified Linear with Gaussian pre-activation) function
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
            A variable holding an array representing the variance of a layer with OPLU units and independent normally distributed pre-activations
    """
    return OpluVar().apply((pre_m, pre_v))[0]

def oplu_moments(pre_m, pre_v):
    """ OPLU moments function
    The function calls the functions that compute the mean and variance of the rectified Normal distribution with mean
    ``pre_m`` and variance ``pre_v``.
    Args ``pre_m`` and ``pre_v`` must have the same dimensions (diagonal var.
    matrix) OR ``pre_v`` has one more dimension than ``pre_m`` (var. matrix with
    non-diagonal elements).
    Args:
        pre_m (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        pre_v (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean of a layer with OPLU units and independent normally distributed pre-activations
        ~chainer.Variable:
            A variable holding an array representing the variance of a layer with OPLU units and independent normally distributed pre-activations
    """

    h_m = oplu_mean(pre_m, pre_v)
    h_v, h_cv = oplu_var(pre_m, pre_v)

    return h_m, h_v, h_cv
