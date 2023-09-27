import cupy as cp
import chainer.functions as F

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

def zhen_loss(mean_s, var_s, target, d, *args, **kwargs):
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
    eps = 10**-14
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

def sm(mean_s, var_s, target):
    """ Smoothed margin function

    Args:
        mean_s: array of output mean vector under isotropic Gaussian noise
        var_s: array of output variance vector under isotropic Gaussian noise
        target: array containing the target class output for each input in the minibatch

    Returns: closed form estimate of smoothed margin

    """
    eps = 10**-14
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s.array
    mean = mean[aux3].reshape((len(target),-1))
    mean_sort_idx = mean.argsort(axis = 1)
    row_idx = cp.repeat(cp.arange(len(target)), mean_sort_idx.shape[1]).reshape((len(target), -1))
    mean = mean_s[aux3].reshape((len(target),-1))
    var = var_s[aux3].reshape((len(target), -1))
    mean_sort = mean[row_idx, mean_sort_idx]
    var_sort = var[row_idx, mean_sort_idx]
    c_mean = mean_s[idx_aux, target.array]
    max_mean = mean_sort[:, -1]
    max_var = var_sort[:, -1]
    ru_mean = mean_sort[:, -2]
    ru_var = var_sort[:, -2]
    t1 = F.sqrt((max_var+ru_var)/(2*cp.pi)) * F.exp(-(max_mean-ru_mean)**2/(2*(max_var+ru_var) + eps))
    t2 = ru_mean + (max_mean - ru_mean) * (1 + F.erf((max_mean-ru_mean)/(F.sqrt(2*(max_var+ru_var)) + eps))) * 0.5
    return -(c_mean - (t1 + t2))

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

