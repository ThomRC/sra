import warnings
warnings.filterwarnings("ignore")

import scipy.integrate as integrate
import scipy.special as special
import numpy as np
np.random.seed(0)

""" Functions required for the numerical integrations """

def corr_prob_func(x, mean, var, mean_c, var_c):
    """ Function to calculate the classification probability by numerical integration  """
    return np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - mean_c)**2/(2*var_c))/np.sqrt(2 * np.pi * var_c)

def mean_max_notc_func(x, mean, var, index = 0):
    """ Function to calculate the mean max output channel other than the correct output by numerical integration
    Required to calculate the smoothed margin
    """

    current_mean = mean[index]
    current_var = var[index]
    mean = np.delete(mean, index)
    var = np.delete(var, index)
    return x * np.prod((1/2)*(1 + special.erf((x - mean)/np.sqrt(2 * var)))) * np.exp(-(x - current_mean)**2/(2*current_var))/np.sqrt(2 * np.pi * current_var)

def smoothed_margin(args):
    """ Calculates the smoothed margin by numerical integration """
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
    """ Calculates the correct and runner up outputs probabilities by numerical integration """
    mean = args[0]
    var = args[1]
    target = args[2]
    n_sigma = args[3]
    idx_aux = np.arange(len(target))
    mean_c = mean[idx_aux, target] # array with the mean of correct output for each input sample
    var_c = var[idx_aux, target] # array with the variance of correct output for each input sample
    aux2 = np.zeros_like(mean, dtype = 'bool')
    aux2[idx_aux,target] = True
    aux3 = np.logical_not(aux2)
    mean_cout = mean[aux3].reshape((len(target),-1)) # array with means of all outputs except the correct one
    var_cout = var[aux3].reshape((len(target),-1)) # array with variances of all outputs except the correct one

    aux = np.absolute(mean.max(1)) + np.absolute(mean.min(1))  # maximum difference between the largest and other outputs
    aux2 = np.zeros_like(mean)
    aux2[idx_aux, target] = aux
    not_tgt_out = mean - aux2  # subtracts the only the values of the correct class outputs from the mean matrix by the maximum differece to make the correct output smaller than or equal to the smallest output
    ru_idx = not_tgt_out.argmax(axis=1) # indices of the runner up output by obtaining the argmax of the previous line matrix

    mean_ru = mean[idx_aux,ru_idx] # array with the mean of runner up output for each input sample
    var_ru = var[idx_aux,ru_idx] # array with the variance of runner up output for each input sample
    aux2 = np.zeros_like(mean, dtype='bool')
    aux2[idx_aux, ru_idx] = True
    aux3 = np.logical_not(aux2)
    mean_ruout = mean[aux3].reshape((len(target), -1)) # array with means of all outputs except the runner up one
    var_ruout = var[aux3].reshape((len(target), -1)) # array with variances of all outputs except the runner up one
    c_prob = []
    ru_prob = []
    samples = len(target)

    for i in range(samples):
        # Numerical integration of correct class probability and runner up class probability
        sigma_c = np.sqrt(var_c[i])
        sigma_ru = np.sqrt(var_ru[i])
        c_prob.append(integrate.quad(corr_prob_func, -n_sigma * sigma_c + mean_c[i], n_sigma * sigma_c + mean_c[i], args = (mean_cout[i,:], var_cout[i,:], mean_c[i], var_c[i]))[0])
        ru_prob.append(integrate.quad(corr_prob_func, -n_sigma * sigma_ru + mean_ru[i], n_sigma * sigma_ru + mean_ru[i], args=(mean_ruout[i, :], var_ruout[i, :], mean_ru[i], var_ru[i]))[0])
    return c_prob, ru_prob

