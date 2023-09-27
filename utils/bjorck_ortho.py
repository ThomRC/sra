import typing as tp  # NOQA

import cupy as cp
import chainer
from chainer import Variable, no_backprop_mode
import chainer.functions as F

def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """
    if iters == 0:
        w = w
    elif order == 1:
        for _ in range(iters):
            w = (1 + beta) * w - beta * F.matmul(w,F.matmul(w.T,w))
    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w

def min_ortho_iter(w, beta=0.5, iters=30, order=1):
    """ Function that reduces the number of iterations required to still achieve a mean pairwise dot product between the weight matrix rows lower than 10**-8

    Args:
        w: weight matrix before the ortogonalization
        beta: BO hyperparameter, original value used in all experiments is 0.5
        iters: BO hyperparameter, original value used in all experiments is 30. This value changes during training using the current function
        order: BO hyperparameter, original value used in all experiments is 1

    Returns: the new number of iterations for the current layer

    """
    scaling = get_safe_bjorck_scaling(w)
    mean_dot = 0
    with no_backprop_mode():
        while mean_dot <= 10**-8 and iters > 1:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w.T,beta=beta,iters=iters,order=order).array.T
            aux = ortho_w.array@ortho_w.array.T
            mean_dot = cp.mean(cp.absolute(aux[cp.logical_not(cp.identity(aux.shape[0], dtype = 'bool'))]))
            print(iters)
            print("Mean pairwise dot: ", mean_dot)
            iters -= 1

        iters += 1

        while mean_dot > 10**-8:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w.T,beta=beta,iters=iters,order=order).array.T
            aux = ortho_w.array@ortho_w.array.T
            mean_dot = cp.mean(cp.absolute(aux[cp.logical_not(cp.identity(aux.shape[0], dtype = 'bool'))]))
            print(iters)
            print("Mean pairwise dot: ", mean_dot)
            iters += 1

        iters -= 1

        return iters

def get_safe_bjorck_scaling(weight):
    """ Gets the current layer scaling factor to guarantee convergence for BO """
    bjorck_scaling = chainer.Variable(cp.sqrt(weight.shape[0] * weight.shape[1]).astype(dtype=cp.float32))

    return bjorck_scaling
