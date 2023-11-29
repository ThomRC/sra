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

def bjorck_orthonormalize_bcop(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if w.shape[-2] < w.shape[-1]:
        return F.moveaxis(bjorck_orthonormalize(
            F.moveaxis(w, -1, -2),
            beta=beta, iters=iters, order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling),(
            -1, -2))

    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w = (1 + beta) * w - beta * w @ F.moveaxis(w, -1, -2) @ w

    return w


def min_ortho_iter(w, beta=0.5, iters=30, order=1, first = False):
    """ Function that reduces the number of iterations required to still achieve a mean pairwise dot product between the weight matrix rows lower than 10**-8

    Args:
        w: weight matrix before the ortogonalization
        beta: BO hyperparameter, original value used in all experiments is 0.5
        iters: BO hyperparameter, original value used in all experiments is 30. This value changes during training using the current function
        order: BO hyperparameter, original value used in all experiments is 1

    Returns: the new number of iterations for the current layer

    """
    scaling = get_safe_bjorck_scaling(w)
    mean_dot_offdiag = 0.
    mean_dot_diag = 1.
    with no_backprop_mode():
        eye = cp.identity(w.array.shape[0], dtype = 'bool')
        not_eye = cp.logical_not(eye)
        while mean_dot_offdiag <= 10**-8 and mean_dot_diag >= 0.99999999  and iters > 1:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w.T,beta=beta,iters=iters,order=order).array.T
            aux = ortho_w.array@ortho_w.array.T
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_eye]))
            mean_dot_diag = cp.mean(cp.absolute(aux[eye]))
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters -= 1

        iters += 1

        while mean_dot_offdiag > 10**-8 or mean_dot_diag < 0.99999999:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w.T,beta=beta,iters=iters,order=order).array.T
            aux = ortho_w.array@ortho_w.array.T
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_eye]))
            mean_dot_diag = cp.mean(cp.absolute(aux[eye]))
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters += 1

        return iters
        
def min_ortho_iter_bcop(w, beta=0.5, iters=30, order=1, first = False):
    """ Function that reduces the number of iterations required to still achieve a mean pairwise dot product between the weight matrix rows lower than 10**-8

    Args:
        w: weight matrix before the ortogonalization
        beta: BO hyperparameter, original value used in all experiments is 0.5
        iters: BO hyperparameter, original value used in all experiments is 30. This value changes during training using the current function
        order: BO hyperparameter, original value used in all experiments is 1

    Returns: the new number of iterations for the current layer

    """
    scaling = get_safe_bjorck_scaling(w)
    mean_dot_offdiag = 0.
    mean_dot_diag = 1.
    with no_backprop_mode():
        not_diag = cp.ones_like(w.array @ cp.moveaxis(w.array, -1, -2), dtype = 'bool')
        not_diag[:,cp.arange(not_diag.shape[-1]),cp.arange(not_diag.shape[-1])] = False
        diag = cp.logical_not(not_diag)
        while mean_dot_offdiag <= 5*10**-8 and mean_dot_diag >= 0.99999995 and iters > 1:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize_bcop(ortho_w,beta=beta,iters=iters,order=order).array
            aux = ortho_w.array @ cp.moveaxis(ortho_w.array, -1, -2)
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_diag]))
            mean_dot_diag = cp.mean(cp.absolute(aux[diag]))
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters -= 1

        iters += 1

        while mean_dot_offdiag > 5*10**-8 or mean_dot_diag < 0.99999995:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize_bcop(ortho_w,beta=beta,iters=iters,order=order).array
            aux = ortho_w.array @ cp.moveaxis(ortho_w.array, -1, -2)
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_diag]))
            mean_dot_diag = cp.mean(cp.absolute(aux[diag]))        
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters += 1

        return iters

def get_safe_bjorck_scaling(weight):
    """ Gets the current layer scaling factor to guarantee convergence for BO """
    bjorck_scaling = chainer.Variable(cp.sqrt(weight.shape[-1] * weight.shape[-2]).astype(dtype=cp.float32))

    return bjorck_scaling
