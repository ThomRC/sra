import scipy.integrate as integrate
import scipy.special as special
import numpy as np
np.random.seed(0)

def sample_mean_margin(network, val_in, target, sd, n, dist = 'Normal', radius = 0.3):
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    with no_backprop_mode():
        margin_sum = network.model.xp.zeros(10000)
        margin_sq_sum = network.model.xp.zeros(10000)
        for _ in range(n):
            noise = network.model.noise_inj(val_in, sd, dist, radius)
            h = val_in + noise
            for link in range(len(network.model) - 1):
                if hasattr(network.model[link], 'W'):
                    h = activation(network.model[link](h))
                else:
                    h = network.model[link](h)
            h = network.model[link + 1](h)
            aux2 = network.model.xp.zeros_like(h.array, dtype = 'bool')
            aux2[network.model.xp.arange(len(target)),target.array] = True
            aux3 = network.model.xp.logical_not(aux2)
            not_tgt_out = h.array[aux3].reshape((len(target),-1))
            not_tgt_max_idx = not_tgt_out.argmax(axis=1)
            margin = network.model.xp.clip(h.array[network.model.xp.arange(10000),target.array] - not_tgt_out[network.model.xp.arange(10000),not_tgt_max_idx], a_min = 0, a_max = None )
            margin_sum += margin
            margin_sq_sum += margin**2
        return margin_sum/n, (margin_sq_sum/n - (margin_sum/n)**2) * n/(n-1)

def sampleundernoise(network, val_in, sd, n, dist = 'Normal', radius = 0.3):
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    with no_backprop_mode():
        count_mat = network.model.xp.zeros((10000,10))
        for _ in range(n):
            noise = network.model.noise_inj(val_in, sd, dist, radius)
            h = val_in + noise
            for link in range(len(network.model) - 1):
                if hasattr(network.model[link], 'W'):
                    # h = F.relu(network[link](h))
                    h = activation(network.model[link](h))
                else:
                    h = network.model[link](h)
            h = network.model[link + 1](h)
            corr_out_idx = network.model.xp.argmax(h.array, axis = 1)
            count_mat[np.arange(10000),corr_out_idx] += 1
        return count_mat

def corr_prob_func(x, mean, var, mean_c, var_c):
    return np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def corr_prob_ub_func(x, mean_1, var_1, mean_c, var_c):
    return (1/2)*(1 + special.erf((x - mean_1)/np.sqrt(2 * var_1))) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def corr_prob_lb_func(x, mean_1, var_1, mean_c, var_c, classes = 10):
    return ((1/2)*(1 + special.erf((x - mean_1)/np.sqrt(2 * var_1))))**(classes-1) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def corr_prob_lb_n_rlx_func(x, mean_1, var_1, mean_c, var_c, n = 5, classes = 10):
    return ((1/2)*(1 + special.erf((x - mean_1)/np.sqrt(2 * var_1))))**(classes-1-n) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def corr_prob_ub_anal(mean_1, var_1, mean_c, var_c):
    return (1/2)*(1 + special.erf((mean_c - mean_1)/np.sqrt(2 * (var_c + var_1))))

def ru_out_prob_func(x, mean, var, mean_c, var_c):
    # function to calculate the mean of output channel under the condition it is the runner-up output
    ru_idx = np.argmax(mean)
    current_mean = mean[ru_idx]
    current_var = var[ru_idx]
    mean = np.delete(mean, ru_idx)
    var = np.delete(var, ru_idx)
    return (1 - special.erf((x - mean_c)/np.sqrt(2 * var_c)))/2 * np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - current_mean)**2/(2*current_var))/np.sqrt(2 * np.pi * current_var)

def mean_corr_func(x, mean, var, mean_c, var_c):
    # function to calculate the mean of output channel under the condition it is the runner-up output
    return x * np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def mean_ru_out_func(x, mean, var, mean_c, var_c, index = 0):
    # function to calculate the mean output under the condition it is the runner-up output
    current_mean = mean[index]
    current_var = var[index]
    mean = np.delete(mean, index)
    var = np.delete(var, index)
    return x * (1 - special.erf((x - mean_c)/np.sqrt(2 * var_c)))/2 * np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - current_mean)**2/(2*current_var))/np.sqrt(2 * np.pi * current_var)

def mean_max_notc_func(x, mean, var, index = 0):
    # function to calculate the mean max output channel other than the correct output
    current_mean = mean[index]
    current_var = var[index]
    mean = np.delete(mean, index)
    var = np.delete(var, index)
    return x * np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - current_mean)**2/(2*current_var))/np.sqrt(2 * np.pi * current_var)

def smoothed_margin(args):
    mean = args[0]
    var = args[1]
    target = args[2]
    n_sigma = args[3]
    n_classes = mean.shape[1]
    mean_c = mean[np.arange(len(target)), target]#.reshape((-1,1))
    aux2 = np.zeros_like(mean, dtype = 'bool')
    aux2[np.arange(len(target)),target] = True
    aux3 = np.logical_not(aux2)
    mean = mean[aux3].reshape((len(target),-1))
    var = var[aux3].reshape((len(target),-1))
    smooth_margin = []
    mean_max_notc = np.zeros(n_classes - 1)
    samples = len(target)
    for i in range(samples):
        sigma_ru = np.sqrt(var[i,:])
        for j in range(n_classes - 1):
            mean_max_notc[j] = integrate.quad(mean_max_notc_func, -n_sigma*sigma_ru[j] + mean[i,j], n_sigma*sigma_ru[j] + mean[i,j], args = (mean[i,:],var[i,:],j))[0]
        smooth_margin.append(mean_c[i] - np.sum(mean_max_notc))
    return smooth_margin

def corr_probabilities(args):
    mean = args[0]
    var = args[1]
    target = args[2]
    n_sigma = args[3]
    idx_aux = np.arange(len(target))
    mean_c = mean[idx_aux, target]#.reshape((-1,1))
    var_c = var[idx_aux, target]#.reshape((-1,1))
    aux2 = np.zeros_like(mean, dtype = 'bool')
    aux2[idx_aux,target] = True
    aux3 = np.logical_not(aux2)
    mean_cout = mean[aux3].reshape((len(target),-1))
    var_cout = var[aux3].reshape((len(target),-1))

    aux = np.absolute(mean.max(1)) + np.absolute(mean.min(1))  # maximum distance from the largest and smallest output
    aux2 = np.zeros_like(mean)
    aux2[idx_aux, target] = aux
    not_tgt_out = mean - aux2  # subtracts the correct class out by the maximum distance to make the correct output smaller than or equal to the smallest output
    ru_idx = not_tgt_out.argmax(axis=1)

    mean_ru = mean[idx_aux,ru_idx]
    var_ru = var[idx_aux,ru_idx]
    aux2 = np.zeros_like(mean, dtype='bool')
    aux2[idx_aux, ru_idx] = True
    aux3 = np.logical_not(aux2)
    mean_ruout = mean[aux3].reshape((len(target), -1))
    var_ruout = var[aux3].reshape((len(target), -1))
    c_prob = []
    ru_prob = []
    samples = len(target)

    for i in range(samples):
        sigma_c = np.sqrt(var_c[i])
        sigma_ru = np.sqrt(var_ru[i])
        c_prob.append(integrate.quad(corr_prob_func, -n_sigma * sigma_c + mean_c[i], n_sigma * sigma_c + mean_c[i], args = (mean_cout[i,:], var_cout[i,:], mean_c[i], var_c[i]))[0])
        ru_prob.append(integrate.quad(corr_prob_func, -n_sigma * sigma_ru + mean_ru[i], n_sigma * sigma_ru + mean_ru[i], args=(mean_ruout[i, :], var_ruout[i, :], mean_ru[i], var_ru[i]))[0])
    return c_prob, ru_prob

def corr_prob_o1_appr_anal(mean, var, target, classes = 10):
    # products of erf are approximated by a single erf of largest mu
    # by doing this, the product of prod(1 + erf((Oc - mu_i)/(sqrt(2)*sigma_i))) for all becomes 1 + sum(2**(9-i)*erf((mu_c - mu_i)/(sqrt(2*(sigma_c**2 + sigma_i**2)))
    mean_c = mean[np.arange(len(target)), target].reshape((-1,1))
    var_c = var[np.arange(len(target)), target].reshape((-1,1))
    aux2 = np.zeros_like(mean, dtype = 'bool')
    aux2[np.arange(len(target)),target] = True
    aux3 = np.logical_not(aux2)
    mean = mean[aux3].reshape((len(target),-1))
    var = var[aux3].reshape((len(target),-1))
    mult_fac = (2**np.arange(classes-1)).reshape((1,classes-1))
    erf_terms = special.erf((mean_c - mean)/np.sqrt(2*(var_c + var)))
    idx_sort = np.argsort(mean, axis = 1)
    prob_appr = (1 + np.sum(mult_fac * erf_terms[np.arange(mean.shape[0]).reshape((-1,1)),idx_sort], axis = 1))/(2**(classes-1))
    print("Probability of correct class (anal. approx. order 1): ", prob_appr)
    return prob_appr

def smoothed_margin_ub(mean_s, var_s, target):
    eps = 10**-8
    aux =  np.absolute(np.max(mean_s,axis = 1)) + np.absolute(np.min(mean_s,axis = 1)) # maximum distance from the largest and smallest output
    aux2 = np.zeros_like(mean_s)
    aux2[np.arange(len(target)),target] = aux
    not_tgt_mean_s = mean_s - aux2 # subtracts the correct class output by the maximum distance to make the correct output smaller than or equal to the smallest output
    not_tgt_max_idx = not_tgt_mean_s.argmax(axis=1) # obtains the arg of max output other than the correct
    t1 = ((mean_s[np.arange(len(target)),target] - mean_s[np.arange(len(target)),not_tgt_max_idx])/2) * (1 + special.erf((mean_s[np.arange(len(target)),target] - mean_s[np.arange(len(target)),not_tgt_max_idx])/np.sqrt(2*(var_s[np.arange(len(target)),target] + var_s[np.arange(len(target)),not_tgt_max_idx]) + eps)))
    t2 = np.sqrt((var_s[np.arange(len(target)),target] + var_s[np.arange(len(target)),not_tgt_max_idx])/(2*np.pi)) * np.exp(-(mean_s[np.arange(len(target)),target] - mean_s[np.arange(len(target)),not_tgt_max_idx])**2/(2*(var_s[np.arange(len(target)),target] + var_s[np.arange(len(target)),not_tgt_max_idx]) + eps))
    return t1 + t2
