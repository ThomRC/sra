import cupy as cp
import cupyx.scipy.special.erf as erf
import chainer.functions as F
from chainer import function_node
from chainer.utils import type_check
import numpy as np

def sce(logits, foo, target, *args, **kwargs):
    """ Average softmax cross-entropy loss over logits () over minibatch.
    This function

    Args:
        logits: array of output vectors before applying softmax function
        foo: dummy variable so that all losses have as first and second arguments the mean output and output variance.
        Since SCE isn't a smoothed loss it doesn't take the output variance
        target: array containing the target class output for each input in the minibatch
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: mean softmax cross-entropy loss over minibatch

    """
    return F.mean(F.softmax_cross_entropy(logits, target))

def sc_sce(mean_s, var_s, target, *args, **kwargs):
    """ Smoothed classifier SCE loss

    Args:
        mean_s: array of output mean vector under isotropic Gaussian noise
        var_s: array of output variance vector under isotropic Gaussian noise
        target: array containing the target class output for each input in the minibatch
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: SCE loss for the mean output

    """
    return sce(mean_s, var_s, target)

def mc_hinge(logits, foo, target, d, *args, **kwargs):
    """ Multiclass hinge loss

    Args:
        logits: array of output vectors before applying softmax function
        foo: dummy variable so that all losses have as first and second arguments the mean output and output variance.
        Since SCE isn't a smoothed loss it doesn't take the output variance
        target: array containing the target class output for each input in the minibatch
        d: hyperpameter that controls the maximal contribution of a single sample to the loss
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: mean multiclass hinge loss over minibatch

    """
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(logits.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    c_out = F.broadcast_to(logits[idx_aux, target.array].reshape((len(target),-1)), (len(target),logits.shape[1]-1))
    notc_out = logits[aux3].reshape((len(target),-1))

    return F.mean(F.sum(F.relu(d - (c_out - notc_out)), axis = 1))

def zhen_loss(mean_s, var_s, cov_s, target, d, *args, **kwargs):
    """ Zhen (randomized smoothing certified radius) loss

    Args:
        mean_s: array of output mean vector under isotropic Gaussian noise
        var_s: array of output variance vector under isotropic Gaussian noise
        target: array containing the target class output for each input in the minibatch
        d: hyperpameter that controls the maximal contribution of a single sample to the loss
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: mean Zhen loss over minibatch

    """
    x_var = kwargs['x_var']
    eps = 10**-8
    idx_aux = cp.arange(len(target))
    aux =  cp.absolute(mean_s.array.max(1)) + cp.absolute(mean_s.array.min(1)) # maximum distance from the largest and smallest output
    aux2 = cp.zeros_like(mean_s.array)
    aux2[idx_aux,target.array] = aux
    not_tgt_mean_s = mean_s - aux2 # subtracts the correct class output by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_mean_s.array.argmax(axis=1) # obtains the arg of max output other than the correct
    return F.mean(F.relu(d - cp.sqrt(x_var)*(mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx])/(F.sqrt(var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx]) + eps)))

def mcr_loss(logits, foo, target, d, *args, **kwargs):
    """ Margin certified radius loss

    Args:
        logits: array of output vectors before applying softmax function
        foo: dummy variable so that all losses have as first and second arguments the mean output and output variance.
        Since SCE isn't a smoothed loss it doesn't take the output variance
        target: array containing the target class output for each input in the minibatch
        d: hyperpameter that controls the maximal contribution of a single sample to the loss
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: mean margin certified radius loss over minibatch

    """
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(logits.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    notc_out = logits[aux3].reshape((len(target),-1))
    not_tgt_max_idx = notc_out.array.argmax(axis=1)
    return F.mean(F.relu(d - (logits[idx_aux, target.array] - notc_out[idx_aux, not_tgt_max_idx])))

# def sm(mean_s, var_s, target):
#     """ Smoothed margin function

#     Args:
#         mean_s: array of output mean vector under isotropic Gaussian noise
#         var_s: array of output variance vector under isotropic Gaussian noise
#         target: array containing the target class output for each input in the minibatch

#     Returns: closed form estimate of smoothed margin

#     """
#     eps = 10**-14
#     idx_aux = cp.arange(len(target))
#     c_mean = mean_s[idx_aux, target.array]
    
#     aux3 = cp.zeros_like(mean_s.array, dtype = 'bool')
#     aux3[idx_aux,target.array] = True
#     aux3 = cp.logical_not(aux3)
#     mean_not_c = mean_s[aux3].reshape((len(target),-1)) # output means excepet correct output's
#     mean_sort_idx = mean_not_c.array.argsort(axis = 1)
#     row_idx = cp.repeat(cp.arange(len(target)), mean_sort_idx.shape[1]).reshape((len(target), -1))
#     var_not_c = var_s[aux3].reshape((len(target), -1))
    
#     mean_sort = mean_not_c[row_idx, mean_sort_idx]
#     var_sort = var_not_c[row_idx, mean_sort_idx]
#     max_mean = mean_sort[:, -1]
#     max_var = var_sort[:, -1]
#     ru_mean = mean_sort[:, -2]
#     ru_var = var_sort[:, -2]
#     t1 = F.sqrt((max_var+ru_var)/(2*cp.pi)) * F.exp(-(max_mean-ru_mean)**2/(2*(max_var+ru_var) + eps))
#     t2 = ru_mean + (max_mean - ru_mean) * (1 + F.erf((max_mean-ru_mean)/(F.sqrt(2*(max_var+ru_var) + eps)))) * 0.5
#     return -(c_mean - (t1 + t2))


class SmoothMargin(function_node.FunctionNode):

    """Rectified Linear Unit variance (i.e., Rectified normal distribution mean) class"""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('out_c_m', 'out_max_m', 'out_max_v', 'out_ru_m', 'out_ru_v'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        eps = 10**-8 # numerical stability constant
        self.retain_inputs((0, 1, 2, 3, 4))
        c_mean, max_mean, max_var, ru_mean, ru_var = inputs
        # alpha = (max_mean - ru_mean)/(cp.sqrt(2 * (max_var + ru_var) + eps))
        alpha_sq = (max_mean - ru_mean)**2/(2 * (max_var + ru_var) + eps)
        # aux = cp.sqrt((max_var+ru_var)/(2*cp.pi)) * cp.exp(-alpha**2) + ru_mean + (max_mean - ru_mean) * (1 + erf(alpha)) * 0.5
        aux = cp.sqrt((max_var+ru_var)/(2*cp.pi)) * cp.exp(-alpha_sq) + ru_mean + (max_mean - ru_mean) * (1 + erf(cp.sqrt(alpha_sq))) * 0.5
        sm = aux - c_mean
        return sm,

    def backward(self, indexes, grad_outputs):
        eps = 10**-8 # numerical stability constant
        gy, = grad_outputs
        c_mean, max_mean, max_var, ru_mean, ru_var = self.get_retained_inputs()
        max_mean = max_mean.array
        max_var = max_var.array
        ru_mean = ru_mean.array
        ru_var = ru_var.array
        # alpha = (max_mean - ru_mean)/(cp.sqrt(2 * (max_var + ru_var) + eps))
        alpha_sq = (max_mean - ru_mean)**2/(2 * (max_var + ru_var) + eps)

        gmean_c = -gy
        # gmean_max = gy * (1 + erf(alpha)) * 0.5
        gmean_max = gy * (1 + erf(cp.sqrt(alpha_sq))) * 0.5
        # gvar_max = gy * 0.5 * cp.exp(-alpha**2) / cp.sqrt(2 * cp.pi * (max_var + ru_var) + eps)
        gvar_max = gy * 0.5 * cp.exp(-alpha_sq) / cp.sqrt(2 * cp.pi * (max_var + ru_var) + eps)
        gmean_ru = gy - gmean_max
        gvar_ru = gvar_max
        return gmean_c, gmean_max, gvar_max, gmean_ru, gvar_ru

# class RSloss(function_node.FunctionNode):

#     """Rectified Linear Unit variance (i.e., Rectified normal distribution mean) class"""

#     def check_type_forward(self, in_types):
#         type_check._argname(in_types, ('out_c_m', 'out_max_m', 'out_max_v', 'out_ru_m', 'out_ru_v'))
#         type_check.expect(
#             in_types[0].dtype.kind == 'f',
#             in_types[0].dtype == in_types[1].dtype,
#             in_types[0].shape == in_types[1].shape
#         )

#     def forward_cpu(self, inputs):
#         raise NotImplementedError()

#     def forward_gpu(self, inputs):
#         eps = 10**-14 # numerical stability constant
#         self.retain_inputs((0, 1, 2, 3, 4))
#         c_mean, max_mean, max_var, ru_mean, ru_var = inputs
#         alpha = (max_mean - ru_mean)/(cp.sqrt(2 * (max_var + ru_var) + eps))
#         aux = cp.sqrt((max_var+ru_var)/(2*cp.pi)) * cp.exp(-alpha**2) + ru_mean + (max_mean - ru_mean) * (1 + erf(alpha)) * 0.5
#         sm = aux - c_mean
        
#         mean_s, var_s, target, d, *args, **kwargs):

#         x_var = kwargs['x_var']
#         eps = 10**-14
#         idx_aux = cp.arange(len(target))
#         aux =  cp.absolute(mean_s.array.max(1)) + cp.absolute(mean_s.array.min(1)) # maximum distance from the largest and smallest output
#         aux2 = cp.zeros_like(mean_s.array)
#         aux2[idx_aux,target.array] = aux
#         not_tgt_mean_s = mean_s - aux2 # subtracts the correct class output by the maximum distance to make the correct output smaller than or equal to the smallest output
#         not_tgt_max_idx = not_tgt_mean_s.array.argmax(axis=1) # obtains the arg of max output other than the correct        
        
#         return sm,

#     def backward(self, indexes, grad_outputs):
#         eps = 10**-14 # numerical stability constant
#         gy, = grad_outputs
#         c_mean, max_mean, max_var, ru_mean, ru_var = self.get_retained_inputs()
#         max_mean = max_mean.array
#         max_var = max_var.array
#         ru_mean = ru_mean.array
#         ru_var = ru_var.array
#         alpha = (max_mean - ru_mean)/(cp.sqrt(2 * (max_var + ru_var) + eps))

#         gmean_c = -gy
#         gmean_max = gy * (1 + erf(alpha)) * 0.5
#         gvar_max = gy * 0.5 * cp.exp(-alpha**2) / cp.sqrt(2 * cp.pi * (max_var + ru_var) + eps)
#         gmean_ru = gy - gmean_max
#         gvar_ru = gvar_max
#         return gmean_c, gmean_max, gvar_max, gmean_ru, gvar_ru

def sm(out_m, out_v, target):
    """ Mean of Rectified Normal distribution (a.k.a., mean of Rectified Linear with Gaussian pre-activation) function
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
            A variable holding an array representing the mean of a layer with ReLU units and independent normally distributed pre-activations
    """
    idx_aux = cp.arange(len(target))
    c_mean = out_m[idx_aux, target.array]

    aux3 = cp.zeros_like(out_m.array, dtype = 'bool')
    aux3[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux3) # boolean matrix with 0's only in the entries for the correct class output
    mean_not_c = out_m[aux3].reshape((len(target),-1)) # output means excepet correct output's
    var_not_c = out_v[aux3].reshape((len(target), -1))
    mean_sort_idx = mean_not_c.array.argsort(axis = 1) # the sorted idx of each output without correct class
    row_idx = cp.repeat(cp.arange(len(target)), mean_sort_idx.shape[1]).reshape((len(target), -1))
    mean_sort = mean_not_c[row_idx, mean_sort_idx]
    var_sort = var_not_c[row_idx, mean_sort_idx]
    max_mean = mean_sort[:, -1]
    max_var = var_sort[:, -1]
    ru_mean = mean_sort[:, -2]
    ru_var = var_sort[:, -2]
    
    return SmoothMargin().apply((c_mean, max_mean, max_var, ru_mean, ru_var))[0]

def mscr_loss(mean_s, var_s, target, d, *args, **kwargs):
    """ Margin smoothed certified radius loss

    Args:
        mean_s: array of output mean vector under isotropic Gaussian noise
        var_s: array of output variance vector under isotropic Gaussian noise
        target: array containing the target class output for each input in the minibatch
        d: hyperpameter that controls the maximal contribution of a single sample to the loss
        *args: required to receive any other argument, even though not using them
        **kwargs: required to receive any other argument, even though not using them

    Returns: mean margin smoothed certified radius loss over minibatch

    """
    return F.mean(F.relu(d + sm(mean_s, var_s, target)))

def cw_loss(logits, target, cut = 0):
    """ Carlini and Wagner loss (used for PGD adversarial attack)

    Args:
        logits: array of output vectors before applying softmax function
        target: array containing the target class output for each input in the minibatch
        cut: hyperparameter that controls the "strength" of the attack by guaranteeing a minimal difference between the logits. cut = 0 accepts as successful attack when correct and incorrect logits are the same

    Returns: mean Carlini and Wagner loss over minibatch

    """
    idx_aux = cp.arange(len(target))

    aux =  cp.absolute(logits.array.max(1)) + cp.absolute(logits.array.min(1)) # maximum distance from the largest and smallest output
    aux2 = cp.zeros_like(logits.array)
    aux2[idx_aux,target.array] = aux
    not_tgt_out = logits - aux2 # subtracts the correct class out by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_out.array.argmax(axis=1)# obtains the arg of max output other than the correct

    return F.mean(-F.relu(cut + logits[idx_aux,target.array] - not_tgt_out[idx_aux,not_tgt_max_idx]))

