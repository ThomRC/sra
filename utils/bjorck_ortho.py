"""
Code from Qiyang Li just adapted to be used with Chainer and added the min_ortho_iter function:
https://github.com/ColinQiyangLi/LConvNet/blob/master/lconvnet/layers/utils.py

Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger Grosse, Jörn-Henrik Jacobsen. "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks"
33rd Conference on Neural Information Processing Systems (NeurIPS 2019)
"""
import cupy as cp
import chainer
from chainer import Variable, no_backprop_mode
import chainer.functions as F

def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """ Function to orthonormalize a matrix or matrices
    
    Algorithm from:
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if w.shape[-2] < w.shape[-1]:
        return F.moveaxis(bjorck_orthonormalize(F.moveaxis(w, -1, -2),
                            beta=beta, iters=iters, order=order), -1, -2)

    assert order == 1, "Only first order Bjorck is supported"
    for _ in range(iters):
        w = (1 + beta) * w - beta * w @ F.moveaxis(w, -1, -2) @ w

    return w

def min_ortho_iter(w, beta=0.5, iters=30, order=1, first = False):
    """ Function that reduces the number of iterations required to still achieve a mean pairwise dot product 
    between the weight matrix rows <= 10**-7 and the mean row norm is >= 0.999998

    Args:
        w: weight matrix before the ortogonalization
        beta: BO hyperparameter, original value used in all experiments is 0.5
        iters: BO hyperparameter, original value used in all experiments is 30. This value changes during training using the current function
        order: BO hyperparameter, original value used in all experiments is 1

    Returns: the new number of iterations for the current layer
    """
    ths1 = 10**-7
    ths2 = 0.999998
    
    scaling = get_safe_bjorck_scaling(w)
    mean_dot_offdiag = 0.
    mean_dot_diag = 1.
    no_cnvg_count1 = 0 # counter of iterations with same mean_dot_offdiag in sequence
    no_cnvg_count2 = 0 # counter of iterations with same mean_dot_offdiag in sequence

    with no_backprop_mode():
        not_diag = cp.ones_like(w.array @ cp.moveaxis(w.array, -1, -2), dtype = 'bool')
        if len(w.array.shape) > 2:
            not_diag[:,cp.arange(not_diag.shape[-1]),cp.arange(not_diag.shape[-1])] = False
        else:
            not_diag[cp.arange(not_diag.shape[-1]),cp.arange(not_diag.shape[-1])] = False
        diag = cp.logical_not(not_diag)
        
        while mean_dot_offdiag <= ths1 and mean_dot_diag >= ths2 and iters > 1:
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w,beta=beta,iters=iters,order=order).array
            aux = ortho_w.array @ cp.moveaxis(ortho_w.array, -1, -2)
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_diag]))
            mean_dot_diag = cp.mean(cp.absolute(aux[diag]))
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters -= 1

        iters += 1

        while mean_dot_offdiag > ths1 or mean_dot_diag < ths2:
            prev_offdiag = mean_dot_offdiag
            prev_diag = mean_dot_diag
            
            ortho_w = Variable(cp.copy(w.array / scaling.array))
            ortho_w.array = bjorck_orthonormalize(ortho_w,beta=beta,iters=iters,order=order).array
            aux = ortho_w.array @ cp.moveaxis(ortho_w.array, -1, -2)
            mean_dot_offdiag = cp.mean(cp.absolute(aux[not_diag]))
            mean_dot_diag = cp.mean(cp.absolute(aux[diag]))        
            print(iters)
            print("Mean pairwise dot offdiag: ", mean_dot_offdiag)
            print("Mean pairwise dot diag: ", mean_dot_diag)
            iters += 1
            
            if mean_dot_offdiag > ths1 and mean_dot_offdiag - prev_offdiag == 0:
                no_cnvg_count1 += 1
            else:
                no_cnvg_count1 = 0
            if mean_dot_diag < ths2 and mean_dot_diag - prev_diag == 0:                
                no_cnvg_count2 += 1
            else:
                no_cnvg_count2 = 0
            
            if no_cnvg_count1 > 3 or no_cnvg_count2 > 3:
                # In case the mean offdiag or diag inner products don't change for 3 iterations, the loop stops
                iters -= 3
                break

        return iters

def get_safe_bjorck_scaling(weight):
    """ Gets the current layer scaling factor to guarantee convergence for BO """
    bjorck_scaling = chainer.Variable(cp.sqrt(weight.shape[-1] * weight.shape[-2]).astype(dtype=cp.float32))

    return bjorck_scaling
