import cupy as cp
import chainer.functions as F

def sce(logits, foo, target, *args, **kwargs):
    return F.mean(F.softmax_cross_entropy(logits, target))

def cw_loss(logits, target, cut = 0):
    idx_aux = cp.arange(len(target))

    aux =  cp.absolute(logits.array.max(1)) + cp.absolute(logits.array.min(1)) # maximum distance from the largest and smallest output
    aux2 = cp.zeros_like(logits.array)
    aux2[idx_aux,target.array] = aux
    not_tgt_out = logits - aux2 # subtracts the correct class out by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_out.array.argmax(axis=1)# obtains the arg of max output other than the correct

    return F.mean(-F.relu(cut + logits[idx_aux,target.array] - not_tgt_out[idx_aux,not_tgt_max_idx]))

def mc_hinge(y, foo, target, d, *args, **kwargs):
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(y.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    c_out = F.broadcast_to(y[idx_aux, target.array].reshape((len(target),-1)), (len(target),y.shape[1]-1))
    notc_out = y[aux3].reshape((len(target),-1))

    return F.mean(F.sum(F.relu(d - (c_out - notc_out)), axis = 1))

def hinge_loss(y, foo, target, d, *args, **kwargs):
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(y.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    notc_out = y[aux3].reshape((len(target),-1))
    not_tgt_max_idx = notc_out.array.argmax(axis=1)
    return F.mean(F.relu(d - (y[idx_aux, target.array] - notc_out[idx_aux, not_tgt_max_idx])))

def smc_hinge(mean_s, var_s, target, d, *args, **kwargs):
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    mean_c = F.broadcast_to(mean_s[idx_aux, target.array].reshape((len(target),-1)), (len(target),mean_s.shape[1]-1))
    var_c = F.broadcast_to(var_s[idx_aux, target.array].reshape((len(target),-1)), (len(target),var_s.shape[1]-1))
    mean = mean_s[aux3].reshape((len(target),-1))
    var = var_s[aux3].reshape((len(target), -1))
    return F.mean(F.sum((d + mean - mean_c) * (1 + F.erf((d + mean - mean_c)/F.sqrt(2*(var_c + var)))) / 2 + F.sqrt((var_c + var)/(2 * cp.pi)) * F.exp(-(d + mean - mean_c)**2/(2*(var_c + var))), axis = 1))

def zhen_loss(mean_s, var_s, target, d, *args, **kwargs):
    x_var = kwargs['x_var']
    eps = 10**-14
    idx_aux = cp.arange(len(target))
    aux =  cp.absolute(mean_s.array.max(1)) + cp.absolute(mean_s.array.min(1)) # maximum distance from the largest and smallest output
    aux2 = cp.zeros_like(mean_s.array)
    aux2[idx_aux,target.array] = aux
    not_tgt_mean_s = mean_s - aux2 # subtracts the correct class output by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_mean_s.array.argmax(axis=1) # obtains the arg of max output other than the correct
    return F.mean(F.relu(d - cp.sqrt(x_var)*(mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx])/(F.sqrt(var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx]) + eps)))

def sc_sce(mean_s, var_s, target, *args, **kwargs):
    return sce(mean_s, var_s, target)

def snmse(mean_s, var_s, h_clean):
    eps = 10**-14
    ### SMOOTHED NORMALIZED MEAN SQUARE ERROR
    t1_l = var_s + mean_s*(mean_s - 2*h_clean)
    t1_r = 1 + F.erf(mean_s/(F.sqrt(2*var_s) + eps))
    t2_l = F.sqrt(var_s/(2*cp.pi)) * (mean_s - 2*h_clean)
    t2_r = F.exp(-mean_s**2/(2*var_s + eps))
    numer = F.sum(t1_l*t1_r + t2_l*t2_r, axis = 1)
    denom = cp.sum(h_clean**2, axis = 1)

    return 1 + numer/(denom + eps)

def smape(mean_s, var_s, h_clean):
    eps = 10**-14
    t1 = mean_s*(1 - F.erf(mean_s/F.sqrt(2*var_s + eps)))*0.5 + F.sqrt(var_s/(2*cp.pi)) * (2*F.exp(-(h_clean-mean_s)**2/(2*var_s + eps)) - F.exp(-mean_s**2/(2*var_s + eps)))
    t2 = (h_clean - mean_s)*F.erf((h_clean - mean_s)/(F.sqrt(2*var_s) + eps))
    numer = F.sum(t1 + t2, axis = 1)
    denom = cp.sum(cp.absolute(h_clean), axis = 1)

    return numer/(denom + eps)

def SDOT(layer, x_m, x_var, x_clean):
    # h_clean, mean_s, var_s, mean_h, var_h = self.moment_propagation(layer, x_m, x_var, x_clean)
    pass

def scw1_loss(mean_s, var_s, target, *args, **kwargs):
    eps = 10**-14
    idx_aux = cp.arange(len(target))
    aux =  cp.absolute(mean_s.array.max(1)) + cp.absolute(mean_s.array.min(1)) # maximum distance from the largest and smallest output
    aux2 = cp.zeros_like(mean_s.array)
    aux2[idx_aux,target.array] = aux
    not_tgt_mean_s = mean_s - aux2 # subtracts the correct class output by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_mean_s.array.argmax(axis=1) # obtains the arg of max output other than the correct
    t1 = (mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx]) * (1 + F.erf((mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx])/(F.sqrt(2*(var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx])) + eps))) * 0.5
    t2 = F.sqrt((var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx])/(2*cp.pi))*F.exp(-(mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx])**2/(2*(var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx]) + eps))
    return F.mean(-F.relu(t1 + t2))

def scw2_loss(mean_s, var_s, target, *args, **kwargs):
    eps = 10**-14
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s.array
    mean = mean[aux3].reshape((len(target),-1))
    mean_sort_idx = mean.argsort(axis = 1)
    max_idx = mean_sort_idx[:,-1]
    max_idx += cp.clip(max_idx - target.array + 1, a_min = 0, a_max = 1)
    ru_idx = mean_sort_idx[:,-2]
    ru_idx += cp.clip(ru_idx - target.array + 1, a_min = 0, a_max = 1)
    c_mean = mean_s[idx_aux, target.array]
    c_var = var_s[idx_aux, target.array]
    max_mean = mean_s[idx_aux, max_idx]
    max_var = var_s[idx_aux, max_idx]
    ru_mean = mean_s[idx_aux, ru_idx]
    ru_var = var_s[idx_aux, ru_idx]

    t1 = 2*((max_var / (F.sqrt(2*cp.pi*(max_var+ru_var)) + eps)) * F.exp(-(max_mean-ru_mean)**2/(2*(max_var+ru_var) + eps)) - (max_var / (F.sqrt(2*cp.pi*(max_var+c_var)) + eps)) * F.exp(-(max_mean-c_mean)**2/(2*(max_var+c_var) + eps)))
    t2 = max_mean * (1 + F.erf(((c_mean-ru_mean)*(ru_var + F.sqrt(c_var*(max_var + ru_var) + max_var*ru_var)) - (max_mean - ru_mean) * (max_var + ru_var))/(F.sqrt(2*(c_var+max_var)*(c_var+ru_var)) + eps))) * 0.5
    return F.mean(-F.relu(c_mean - (t1 + t2)))

def sm1(mean_s, var_s, target):
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
    t1 = (max_var / (F.sqrt(2*cp.pi*(max_var+ru_var)) + eps)) * F.exp(-(max_mean-ru_mean)**2/(2*(max_var+ru_var) + eps))
    t2 = max_mean * (1 + F.erf((max_mean-ru_mean)/(F.sqrt(2*(max_var+ru_var)) + eps))) * 0.5
    return -(c_mean - (t1 + t2))

def sm2(mean_s, var_s, target):
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

def sm1_loss(mean_s, var_s, target, *args, **kwargs):
    return F.mean(sm1(mean_s, var_s, target))

def clip_sm1_loss(mean_s, var_s, target, d, *args, **kwargs):
    return F.mean(F.relu(d + sm1(mean_s, var_s, target)))

def mix_sm1_loss(mean_s1, var_s1, target, foo, mean_s2, var_s2, **kwargs):
    return F.mean((sm1(mean_s1, var_s1, target) + sm1(mean_s2, var_s2, target))/2)

def clip_mix_sm1_loss(mean_s1, var_s1, target, d, mean_s2, var_s2, **kwargs):
    return F.mean((F.relu(d + sm1(mean_s1, var_s1, target)) + F.relu(d + sm1(mean_s2, var_s2, target)))/2)

def sm2_loss(mean_s, var_s, target, *args, **kwargs):
    return F.mean(sm2(mean_s, var_s, target))

def clip_sm2_loss(mean_s, var_s, target, d, *args, **kwargs):
    return F.mean(F.relu(d + sm2(mean_s, var_s, target)))

def mix_sm2_loss(mean_s1, var_s1, target, foo, mean_s2, var_s2, **kwargs):
    return F.mean((sm2(mean_s1, var_s1, target) + sm2(mean_s2, var_s2, target))/2)

def clip_mix_sm2_loss(mean_s1, var_s1, target, d, mean_s2, var_s2, **kwargs):
    return F.mean((F.relu(d + sm2(mean_s1, var_s1, target)) + F.relu(d + sm2(mean_s2, var_s2, target)))/2)

def sm_lb_loss(mean_s, var_s, target, *args, **kwargs):
    return F.mean(sm_lb(mean_s, var_s, target))

def clip_sm_lb_loss(mean_s, var_s, target, d, *args, **kwargs):
    return F.mean(F.relu(d + sm_lb(mean_s, var_s, target)))

def mix_sm_lb_loss(mean_s1, var_s1, target, foo, mean_s2, var_s2, **kwargs):
    return F.mean((sm_lb(mean_s1, var_s1, target) + sm_lb(mean_s2, var_s2, target))/2)

def clip_mix_sm_lb_loss(mean_s1, var_s1, target, d, mean_s2, var_s2, **kwargs):
    return F.mean((F.relu(d + sm_lb(mean_s1, var_s1, target)) + F.relu(d + sm_lb(mean_s2, var_s2, target)))/2)

def sm3(mean_s, var_s, target):
    eps = 10**-14
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s.array
    mean = mean[aux3].reshape((len(target),-1))
    mean_sort_idx = mean.argsort(axis = 1)
    max_idx = mean_sort_idx[:,-1]
    max_idx += cp.clip(max_idx - target.array + 1, a_min = 0, a_max = 1)
    ru_idx = mean_sort_idx[:,-2]
    ru_idx += cp.clip(ru_idx - target.array + 1, a_min = 0, a_max = 1)
    trd_idx = mean_sort_idx[:,-3]
    trd_idx += cp.clip(trd_idx - target.array + 1, a_min = 0, a_max = 1)
    c_mean = mean_s[idx_aux, target.array]
    max_mean = mean_s[idx_aux, max_idx]
    max_var = var_s[idx_aux, max_idx]
    ru_mean = mean_s[idx_aux, ru_idx]
    ru_var = var_s[idx_aux, ru_idx]
    trd_mean = mean_s[idx_aux, trd_idx]
    trd_var = var_s[idx_aux, trd_idx]
    t1 = F.sqrt((max_var+ru_var)/2*cp.pi) * F.exp(-(max_mean-ru_mean)**2/(2*(max_var+ru_var) + eps))
    t2 = ru_mean + (max_mean - ru_mean) * (1 + F.erf((max_mean-ru_mean)/(F.sqrt(2*(max_var+ru_var)) + eps))) * 0.5
    t3 = 0.5 * trd_mean * (1 - F.erf((max_mean-trd_mean) / (F.sqrt(2 * (max_var + trd_var)) + eps))) + trd_var/F.sqrt(2 * cp.pi * (max_var + trd_var)) * F.exp(-(max_mean - trd_mean) ** 2/(2 * (max_var + trd_var) + eps))
    return -(c_mean - (t1 + t2 + t3))

def sm_lb(mean_s, var_s, target):
    eps = 10 ** -14
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype='bool')
    aux2[idx_aux, target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s.array
    mean = mean[aux3].reshape((len(target), -1))
    mean_sort_idx = mean.argsort(axis=1)
    row_idx = cp.repeat(cp.arange(len(target)), mean_sort_idx.shape[1]).reshape((len(target), -1))
    mean = mean_s[aux3].reshape((len(target),-1))
    var = var_s[aux3].reshape((len(target), -1))
    mean_sort = mean[row_idx, mean_sort_idx]
    var_sort = var[row_idx, mean_sort_idx]
    c_mean = mean_s[idx_aux, target.array].reshape((len(target), -1))
    max_mean = mean_sort[:, -1].reshape((len(target), -1))
    max_var = var_sort[:, -1].reshape((len(target), -1))
    p_2 = (1 + F.erf((mean_sort[:, -2] - mean_sort[:, -3]) / (F.sqrt(2 * (var_sort[:, -2] + var_sort[:, -3])) + eps))).reshape((len(target),-1))
    ps = 1 + F.erf((mean_sort[:,:-2] - mean_sort[:, -2].reshape((len(target), -1))) / (F.sqrt(2 * (var_sort[:, -2].reshape((len(target), -1)) + var_sort[:, :-2])) + eps))
    p = p_2 + F.sum(ps, axis = 1).reshape((len(target),-1))
    t1 = F.sqrt((max_var + var_sort[:,:-1]) / (2 * cp.pi)) * F.exp(-(max_mean - mean_sort[:,:-1]) ** 2 / (2 * (max_var + var_sort[:,:-1]) + eps))
    t2 = mean_sort[:,:-1] + (max_mean - mean_sort[:,:-1]) * (1 + F.erf((max_mean - mean_sort[:,:-1]) / (F.sqrt(2 * (max_var + var_sort[:,:-1])) + eps))) * 0.5
    return -(c_mean - (F.sum(ps * (t1[:,:-1] + t2[:,:-1]), axis = 1).reshape((len(target),-1)) + p_2 * (t1[:,-1] + t2[:,-1]).reshape((len(target),-1)))/p)

def clip_pc_lb_loss(mean_s, var_s, target, d, *args, **kwargs):
    eps = 10 ** -7
    x_var = kwargs['x_var']
    return F.mean(F.relu(d - cp.sqrt(2 * x_var) * F.erfinv(pc_lb(mean_s, var_s, target) - eps)))


def pc_lb(mean_s, var_s, target):
    # PC: probability of correct classification
    # lb: lower bound
    eps = 10 ** -14
    n_classes = mean_s.shape[1]
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype='bool')
    aux2[idx_aux, target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s.array
    mean = mean[aux3].reshape((len(target), -1))
    mean_sort_idx = mean.argsort(axis=1)
    row_idx = cp.repeat(cp.arange(len(target)), mean_sort_idx.shape[1]).reshape((len(target), -1))
    mean = mean_s[aux3].reshape((len(target),-1))
    var = var_s[aux3].reshape((len(target), -1))
    mean_sort = mean[row_idx, mean_sort_idx]
    var_sort = var[row_idx, mean_sort_idx]
    c_mean = mean_s[idx_aux, target.array].reshape((len(target), -1))
    c_var = var_s[idx_aux, target.array].reshape((len(target), -1))
    return (1 + F.sum(2 ** cp.arange(n_classes-1).reshape((1, n_classes-1)) * F.erf((c_mean - mean_sort)/(F.sqrt(2 * (c_var + var_sort)) + eps)), axis = 1))/ 2**(n_classes - 1)

def clip_pc_sf_loss(mean_s, var_s, target, d, *args, **kwargs):
    eps = 10 ** -7
    x_var = kwargs['x_var']
    # return F.mean(F.relu(d - pc_lb(mean_s, var_s, target)))
    return F.mean(F.relu(d - cp.sqrt(2 * x_var) * F.erfinv(2 * pc_sf(mean_s, var_s, target) - 1 + eps)))
    # return F.mean(F.relu(d - cp.sqrt(x_var)*(mean_s[idx_aux,target.array] - mean_s[idx_aux,not_tgt_max_idx])/(F.sqrt(var_s[idx_aux,target.array] + var_s[idx_aux,not_tgt_max_idx]) + eps)))

def pc_sf(mean_s, var_s, target):
    eps = 10 ** -14
    sigma_s = cp.sqrt(cp.pi/3)
    n_classes = mean_s.shape[1]
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(mean_s.array, dtype='bool')
    aux2[idx_aux, target.array] = True
    aux3 = cp.logical_not(aux2)
    mean = mean_s[aux3].reshape((len(target),-1))
    var = var_s[aux3].reshape((len(target), -1))
    c_mean = mean_s[idx_aux, target.array].reshape((len(target), -1))
    c_var = var_s[idx_aux, target.array].reshape((len(target), -1))
    return 1/(1 + F.sum(F.exp(sigma_s * (mean - c_mean)/(F.sqrt(c_var + var) + eps)), axis = 1))
