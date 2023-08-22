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

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = F.matmul(w.T,w)
            w_t_w_w_t_w = F.matmul(w_t_w,w_t_w)
            w = (+ (15 / 8) * w
                - (5 / 4) * F.matmul(w, w_t_w)
                + (3 / 8) * F.matmul(w, w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = F.matmul(w.T,w)
            w_t_w_w_t_w = F.matmul(w_t_w,w_t_w)
            w_t_w_w_t_w_w_t_w = F.matmul(w_t_w,w_t_w_w_t_w)
            w = (+ (35 / 16) * w
                 - (35 / 16) * F.matmul(w,w_t_w)
                 + (21 / 16) * F.matmul(w,w_t_w_w_t_w)
                 - (5 / 16) * F.matmul(w,w_t_w_w_t_w_w_t_w))

    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in range(iters):
            w_t_w = F.matmul(w.T,w)
            w_t_w_w_t_w = F.matmul(w_t_w,w_t_w)
            w_t_w_w_t_w_w_t_w = F.matmul(w_t_w,w_t_w_w_t_w)
            w_t_w_w_t_w_w_t_w_w_t_w = F.matmul(w_t_w,w_t_w_w_t_w_w_t_w)

            w = (+ (315 / 128) * w
                 - (105 / 32) * F.matmul(w,w_t_w)
                 + (189 / 64) * F.matmul(w,w_t_w_w_t_w)
                 - (45 / 32) * F.matmul(w,w_t_w_w_t_w_w_t_w)
                 + (35 / 128) * F.matmul(w,w_t_w_w_t_w_w_t_w_w_t_w))

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w

def min_ortho_iter(w, beta=0.5, iters=30, order=1):
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
    bjorck_scaling = chainer.Variable(cp.sqrt(weight.shape[0] * weight.shape[1]).astype(dtype=cp.float32))

    return bjorck_scaling
