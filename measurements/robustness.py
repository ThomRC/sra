import os
import time

from chainer import cuda, no_backprop_mode
import cupy as cp
import numpy as np

import utils.stats_func as utils
from multiprocessing import Pool

from measurements.noise_robustness import gaussian_noise_acc
from measurements.adversarial_attack import linf_pgd, l2_pgd
from measurements.robustness_certification import cohen_cr, lip_cr
from measurements.other import ssnr_eval_acc
from measurements.utils import save_measurements

def adversarial_eval(network, x, target, name_str):
    """
    Empirical adversarial robustness evaluations (L2 and Linf PGD attacks)

    Args:
        network: NNAgent object containing the model subject to the perturbation
        x: input that will be perturbed
        target: correct class array
        name_str: string containing the name code for current net to check if the measurements were already taken

    Returns: the measured accuracies for the four different types of perturbation
    """
    start_time = time.perf_counter()
    print("### 1.1) Adversarial robustness tests (L2 and Linf PGD attacks) ###")

    adv_acc_linf = []
    noise_adv_acc_linf = []
    adv_acc_l2 = []
    noise_adv_acc_l2 = []

    acc = cuda.to_cpu(network.model.validation(x, target, train=False))
    acc_noise = cuda.to_cpu(network.model.validation(x, target, noise_in=True, sd=0.1, train=False))
    print("Clean accuracy : {}".format(acc))
    print("Clean noise accuracy : {}".format(acc_noise))

    adv_acc_linf.append(acc)
    noise_adv_acc_linf.append(acc_noise)
    if not os.path.isfile(name_str + 'linf_adv_acc.npy'):
        aux1, aux2 = linf_pgd(network, x, target)
        adv_acc_linf.extend(cuda.to_cpu(cp.asarray(aux1)))
        noise_adv_acc_linf.extend(cuda.to_cpu(cp.asarray(aux2)))
    else:
        adv_acc_linf = None
        noise_adv_acc_linf = None

    adv_acc_l2.append(acc)
    noise_adv_acc_l2.append(acc_noise)
    if not os.path.isfile(name_str + 'l2_adv_acc.npy'):
        aux1, aux2 = l2_pgd(network, x, target)
        adv_acc_l2.extend(cuda.to_cpu(cp.asarray(aux1)))
        noise_adv_acc_l2.extend(cuda.to_cpu(cp.asarray(aux2)))
    else:
        adv_acc_l2 = None
        noise_adv_acc_l2 = None

    print("Exe. time = {}".format(time.perf_counter() - start_time))
    return adv_acc_linf, noise_adv_acc_linf, adv_acc_l2, noise_adv_acc_l2

def other_eval(network, x, target, name_str):
    """
    Empirical external perturbation evaluations (Gaussian noise perturbation and SSNR linear corruption)

    Args:
        network: NNAgent object containing the model subject to the perturbation
        x: input that will be perturbed
        target: correct class array
        name_str: string containing the name code for current net to check if the measurements were already taken

    Returns: the measured accuracies for the four different types of perturbation
    """
    start_time = time.perf_counter()
    print("### 1.2) Gaussian noise perturbation and SSNR linear corruption robustness tests ###")

    noise_acc = []
    ssnr_acc = []

    acc = cuda.to_cpu(network.model.validation(x, target, train=False))
    print("Clean accuracy : {}".format(acc))
    ssnr_acc.append(acc)
    print(name_str + 'SSNR_acc.npy')
    if not os.path.isfile(name_str + 'SSNR_acc.npy'):
        aux1 = cuda.to_cpu(cp.asarray(ssnr_eval_acc(network, x, target)))
        ssnr_acc.extend(aux1)
    else:
        ssnr_acc = None

    noise_acc.append(acc)
    print(name_str + 'gauss_acc.npy')
    if not os.path.isfile(name_str + 'gauss_acc.npy'):
        aux1 = cuda.to_cpu(cp.asarray(gaussian_noise_acc(network, x, target)))
        noise_acc.extend(aux1)
    else:
        noise_acc = None

    print("Exe. time = {}".format(time.perf_counter() - start_time))
    return noise_acc, ssnr_acc

def numerical_eval(network, x, target):
    """
    Obtain through numerical integration the values required to certify the robustness for different input sd

    Args:
        network: NNAgent object containing the model subject to the perturbation
        x: input that will be perturbed
        target: correct class array

    Returns: correct classification probability, runner up probability, smoothed margin, output layer mean, output layer variance
    """
    t1 = time.perf_counter()
    print("### 2) Numerical integration of output metrics (classification probability and smoothed margin) ###")
    # The first element of the lists are for sd = 0
    _, mean_s, var_s, _, _ = network.model.moment_propagation(len(network.model), x, 0)
    sort_mean = cp.sort(mean_s.array, axis = 1)
    margin_zero = cuda.to_cpu(sort_mean[:,-1] - sort_mean[:,-2]).reshape((1,-1))
    mean_out_arr = []
    var_out_arr = []
    mean_out_arr.append(cuda.to_cpu(mean_s.array))
    var_out_arr.append(cuda.to_cpu(cp.zeros_like(mean_s.array)))
    function_params = []
    for sd in np.linspace(0.03, 1.01, 15):
        with no_backprop_mode():
            _, mean_s, var_s, _, _ = network.model.moment_propagation(len(network.model), x, sd ** 2)

        mean = cuda.to_cpu(mean_s.array)
        var = cuda.to_cpu(var_s.array)
        mean_out_arr.append(mean)
        var_out_arr.append(var)
        aux = [mean, var, target, 4]
        function_params.append(aux)

    p = Pool(32)
    results_pooled1 = p.map(utils.corr_probabilities,function_params)
    results_pooled2 = p.map(utils.smoothed_margin, function_params)
    aux = np.asarray(results_pooled1)

    c_prob = np.vstack((np.ones((1,aux.shape[2])),aux[:,0,:]))
    ru_prob = np.vstack((np.zeros((1,aux.shape[2])),aux[:,1,:]))
    smooth_margin = np.vstack((margin_zero,np.asarray(results_pooled2)))

    print("Exe. time  = {}".format(sd, time.perf_counter() - t1))

    return c_prob, ru_prob, smooth_margin, np.asarray(mean_out_arr), np.asarray(var_out_arr)

def emp_evals(network, x, target):
    """
    Calculates sample estimates of output metrics (classification probability and smoothed margin)

    Args:
        network: NNAgent object containing the model subject to the perturbation
        x: input that will be perturbed
        target: correct class array

    Returns: correct classification probability, runner up probability, smoothed margin, output layer mean, output layer variance
    """
    start_time = time.perf_counter()
    prob_c_arr = []
    margin_mean_arr = []
    margin_var_arr = []

    print("### 3) Sample estimate of output metrics (classification probability and smoothed margin) ###")
    samples_arr = [1, 10, 100, 1000, 10000]
    for sd in np.linspace(0.03, 1.01, 15):
        prob_c_arr2 = []
        margin_mean_arr2 = []
        margin_var_arr2 = []
        for samples in samples_arr:
            t1 = time.perf_counter()
            prob_c_arr2.append(network.model.emp_prob(x, target, sd, samples))
            margin_mean, margin_var = network.model.sample_mean_margin(x, target, sd, samples)
            margin_mean_arr2.append(margin_mean)
            margin_var_arr2.append(margin_var)
            t2 = time.perf_counter()
            print("Exe. time for sd {} and {} samples = {}".format(sd, samples, t2 - t1))
        prob_c_arr.append(prob_c_arr2)
        margin_mean_arr.append(margin_mean_arr2)
        margin_var_arr.append(margin_var_arr2)

    print("Exe. time = {}".format(time.perf_counter() - start_time))
    return cuda.to_cpu(cp.asarray(prob_c_arr)), cuda.to_cpu(cp.asarray(margin_mean_arr)), cuda.to_cpu(cp.asarray(margin_var_arr))

def cr_eval(network, x, target, name_str):
    """
    Evaluation of certified radius

    Args:
        network: NNAgent object containing the model subject to the perturbation
        x: input that will be perturbed
        target: correct class array
        name_str: string containing the name code for current net to check if the measurements were already taken

    Returns: the measured accuracies for the four different types of perturbation
    """
    start_time = time.perf_counter()
    print("### 4) Randomized smoothing (Cohen et al., 2019) and Lipschitz margin certified radii ###")
    
    rs_cr = []
    margin_cr = []

    if not os.path.isfile(name_str + 'lip_cr.npy'):
        aux1 = lip_cr(network, x, target)
        margin_cr.extend(aux1)
    else:
        print("Skipping margin radius certification")
        margin_cr = None
        
    if not os.path.isfile(name_str + 'rs_cr.npy'):
        print(name_str + 'rs_cr.npy')
        aux1 = cohen_cr(network, x, target)
        rs_cr.extend(aux1)
    else:
        print("Skipping RS radius certification")
        rs_cr = None
        
    print("Exe. time = {}".format(time.perf_counter() - start_time))
    return rs_cr, margin_cr

def measurements(network, x, target, dest, robustness = True, num_int = True, sample_est = True, rs_cr = True):
    """
    Calls functions to carry all robustness measurements for the currently loaded NN and stores it into a dictionary that will be used to save the data into .npy files

    Args:
        network: NNAgent object containing the model subject to the pertubation
        x: input that will be perturbed
        target: correct class array
        dest: destination folder to
    """
    print("##### Robustness measurements for Loss: {}; d: {}; x_var: {}; training num.: {}".format(network.loss, network.d, network.x_var, network.model.training_num))
    # Creates name string containing the loss from currently loaded NN, the total trianing epochs, x_var and d hyperparameters, and the training number in case of having more than one trained NN with same settings
    name_str = dest + 'loss_{}_ep_{}_x_var_{}_d_{}_{}_'.format(network.loss, network.epochs, network.x_var, network.d, network.model.training_num)

    t = cuda.to_cpu(target.array)
    _, mean_s, var_s, mean_h, var_h = network.model.moment_propagation(len(network.model), x, 0)
    ## Obtain the metrics only for correctly classified samples
    ## For this we need a boolean ndarray of correct samples
    corr_class = cuda.to_cpu(cp.argmax(mean_s.array, axis=1)) # ndarray containing the output indices with maximal value
    corr_idx = corr_class == t # boolean ndarray of size = the number of test samples with True where correctly classified
    corr_samples = np.arange(10000)[corr_idx]  # ndarray containing the indices of correctly classified samples
    in_shape = [i for i in x.shape[1:]]
    data_shape = [len(corr_samples)]
    data_shape.extend(in_shape)
    network.model.data_shape = data_shape
    corr_x = x[corr_samples,:].reshape(data_shape) # ndarray containing the test samples that are correclt classified
    corr_t = target[corr_idx] # ndarray containing the labels of correctly classified inputs

    measurements_dict = {}
    measurements_dict['target'] = cuda.to_cpu(corr_t.array)
    measurements_dict['clean_acc'] = len(corr_t)/len(t)
    
    if robustness:
        # measurements_dict['linf_adv_acc'], measurements_dict['linf_adv_acc_noise'], measurements_dict['l2_adv_acc'], measurements_dict['l2_adv_acc_noise'] = adversarial_eval(network, corr_x, corr_t, name_str)
        measurements_dict['gauss_acc'], measurements_dict['SSNR_acc'] = other_eval(network, x, target, name_str)

    if num_int:
        if any(not os.path.isfile(name_str + measurement) for measurement in ['p_c.npy', 'p_ru.npy', 'smooth_margin.npy', 'mean_out.npy', 'var_out.npy']):
            measurements_dict['p_c'], measurements_dict['p_ru'], measurements_dict['smooth_margin'], measurements_dict['mean_out'], measurements_dict['var_out'] = numerical_eval(network, corr_x, cuda.to_cpu(corr_t.array))
        else:   
            print("Files found for current settings. Skipping the numerical integration of classification probability and smoothed margin.")

    if sample_est:
        if any(not os.path.isfile(name_str + measurement) for measurement in ['emp_p_c.npy', 'emp_margin_mean.npy', 'emp_margin_var.npy']):
            measurements_dict['emp_p_c'], measurements_dict['emp_margin_mean'], measurements_dict['emp_margin_var'] = emp_evals(network, x, target)
        else:
            print("Files found for current settings. Skipping the sample estimate of classification probability and smoothed margin.")

    if rs_cr:
        if any(not os.path.isfile(name_str + measurement) for measurement in ['rs_cr.npy','lip_cr.npy']):
            measurements_dict['rs_cr'], measurements_dict['lip_cr'] = cr_eval(network, x, target, name_str)
        else:   
            print("Files found for current settings. Skipping the randomized smoothing radius certification.")
        
    save_measurements(name_str, measurements_dict)
