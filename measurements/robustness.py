import os
import glob
import time
import code

from chainer import cuda, Variable, no_backprop_mode
import cupy as cp
import numpy as np

import utils.stats_func as utils
from multiprocessing import Pool

def linf_attacks(network, x, target):
    """ Carries linf-norm PGD adversarial attacks with with gradual increments of adversarial radius until all samples are successfully attacked to measure robustness against linf attacks

    Args:
        network: NNAgent object containing the model subject to attacks
        x: input that will be perturbed
        target: correct class array

    Returns: adversarial accuracy and adversarial accuracy under noise for each attack radius

    """
    if network.dataset == "MNIST":
        radius_step = 0.025
    elif network.dataset == "cifar10":
        radius_step = 0.00625
    scale = 1.0
    cw_cut = 0
    loss_func = 'CW'  ## 'crossentropy', 'CW', 'sm'
    num_steps = 200
    step_norm = 'sign'  ## 'sign', 'inf', '2'
    eps_norm = 'inf'  ## 'inf', '2'
    adv_acc = []
    noise_adv_acc = []
    samples = 5
    radius = 0
    aux = cp.zeros(samples)
    counter = 0
    while radius == 0 or cp.min(aux) > 0:
        radius += radius_step
        aux = cp.zeros(samples)
        aux2 = cp.zeros(samples)
        for i in range(samples):
            step_size = 0.08 * radius
            t1_1 = time.perf_counter()
            x_adv = network.model.projected_gradient_descent(x, target, num_steps=num_steps,
                                                             step_size=step_size, eps=radius, clamp=(
                    network.center_in - network.intvl_in / 2, network.center_in + network.intvl_in / 2), eps_norm=eps_norm,
                                                             step_norm=step_norm, loss_func=loss_func, pert_scale=scale,
                                                             cw_cut=cw_cut)
            aux[i] = network.model.validation(Variable(x_adv), target, noise_in=False, train=False)
            print("Accuracy for Linf radius = {} is {} #{}".format(radius, aux[i], i))
            aux2[i] = network.model.validation(Variable(x_adv), target, noise_in=True, sd=0.1, train=False)
            print("Noise accuracy for Linf radius = {} is {} #{}".format(radius, aux2[i], i))
            t2_1 = time.perf_counter()
            print("Exe. time = {}".format(t2_1 - t1_1))

        adv_acc.append(cp.asarray(aux).min())
        noise_adv_acc.append(cp.asarray(aux2).min())
        if cp.min(aux) < 0.001:
            counter += 1
        
        if counter > 3:
            print("Some samples wont converge. Stopping Linf attacks earlier")
            break
    return adv_acc, noise_adv_acc

def l2_attacks(network, x, target):
    """ Carries l2-norm PGD adversarial attacks with with gradual increments of adversarial radius until all samples are successfully attacked to measure robustness against l2 attacks

    Args:
        network: NNAgent object containing the model subject to attacks
        x: input that will be perturbed
        target: correct class array

    Returns: adversarial accuracy and adversarial accuracy under noise for each attack radius

    """
    if network.dataset == "MNIST":
        radius_step = 0.32
    elif network.dataset == "cifar10":
        radius_step = 0.16
    scale = 1.0
    cw_cut = 0
    loss_func = 'CW'  ## 'crossentropy', 'CW', 'sm'
    num_steps = 200
    step_norm = '2'  ## 'sign', 'inf', '2'
    eps_norm = '2'  ## 'inf', '2'
    adv_acc = []
    noise_adv_acc = []
    samples = 5
    radius = 0
    aux = cp.zeros(samples)
    counter = 0
    while radius == 0 or cp.min(aux) > 0:
        radius += radius_step
        aux = cp.zeros(samples)
        aux2 = cp.zeros(samples)
        for i in range(samples):
            step_size = 0.08 * radius
            t1_1 = time.perf_counter()
            x_adv = network.model.projected_gradient_descent(x, target, num_steps=num_steps,
                                                             step_size=step_size, eps=radius, clamp=(
                    network.center_in - network.intvl_in / 2, network.center_in + network.intvl_in / 2), eps_norm=eps_norm,
                                                             step_norm=step_norm, loss_func=loss_func, pert_scale=scale,
                                                             cw_cut=cw_cut)
            aux[i] = network.model.validation(Variable(x_adv), target, noise_in=False, train=False)
            print("Accuracy for L2 radius = {} is {} #{}".format(radius, aux[i], i))
            aux2[i] = network.model.validation(Variable(x_adv), target, noise_in=True, sd=0.1, train=False)
            print("Noise accuracy for L2 radius = {} is {} #{}".format(radius, aux2[i], i))
            t2_1 = time.perf_counter()
            print("Exe. time = {}".format(t2_1 - t1_1))

        adv_acc.append(cp.asarray(aux).min())
        noise_adv_acc.append(cp.asarray(aux2).min())
        if cp.min(aux) < 0.001:
            counter += 1
        
        if counter > 3:
            print("Some samples wont converge. Stopping L2 attacks earlier")
            break
    return adv_acc, noise_adv_acc

def gaussian_noise_acc(network, x, target):
    """ Measures the accuracy under Gaussian noise with with gradual increments of noise variance until average accuracy reaches 1/c, where c is the number of output classes

    Args:
        network: NNAgent object containing the model subject to the pertubation
        x: input that will be perturbed
        target: correct class array

    Returns: accuracy under noise for each input variance

    """
    if network.dataset == "MNIST":
        acc_threshold = 0.13
        sd_step = 0.32
    elif network.dataset == "cifar10":
        acc_threshold = 0.13
        sd_step = 0.16
    elif network.dataset == "cifar100":
        acc_threshold = 0.013
        sd_step = 0.16

    noise_acc = []
    samples = 5
    sd = 0
    aux = cp.zeros(samples)
    while sd == 0 or cp.mean(aux) > acc_threshold and sd < 10:
        sd += sd_step
        aux = cp.zeros(samples)
        for i in range(samples):
            aux[i] = network.model.validation(x, target, noise_in=True, sd=sd, train=False)
            print("Noise accuracy for sd = {} is {} #{}".format(sd, aux[i], i))

        noise_acc.append(cp.asarray(aux).mean())

    return noise_acc

def ssnr_eval_acc(network, x, target):
    """ Measures the accuracy after combining the input image with a "gray" image (all pixels with same value) from clean image (ssnr_ratio = 1.0) to completely corrupted image (ssnr_ratio = 0.0)
    (SSNR = signal-to-signal-plu-noise ratio)

    Args:
        network: NNAgent object containing the model subject to the pertubation
        x: input that will be perturbed
        target: correct class array

    Returns: accuracy under SSNR corruption for different SSNR ratios

    """
    noise_acc = []
    samples = 10
    for ssnr_ratio in np.linspace(0.95, 0, 20):
        aux = cp.zeros(samples)
        for i in range(samples):
            aux[i] = network.model.validation(x, target, noise_in=True, sd=1.0, train=False, ssnr = True, ssnr_ratio = ssnr_ratio)

        print("Accuracy for SSNR = {} is {}".format(ssnr_ratio, cp.asarray(aux).mean()))
        noise_acc.append(cp.asarray(aux).mean())

    return noise_acc

def adversarial_eval(network, x, target, name_str):
    """ Empirical external perturbation evaluations (L2 and Linf PGD attacks, Gaussian noise perturbation) and SSNR linear corruption)

    Args:
        network: NNAgent object containing the model subject to the pertubation
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
        aux1, aux2 = linf_attacks(network, x, target)
        adv_acc_linf.extend(cuda.to_cpu(cp.asarray(aux1)))
        noise_adv_acc_linf.extend(cuda.to_cpu(cp.asarray(aux2)))
    else:
        adv_acc_linf = None
        noise_adv_acc_linf = None

    adv_acc_l2.append(acc)
    noise_adv_acc_l2.append(acc_noise)
    if not os.path.isfile(name_str + 'l2_adv_acc.npy'):
        aux1, aux2 = l2_attacks(network, x, target)
        adv_acc_l2.extend(cuda.to_cpu(cp.asarray(aux1)))
        noise_adv_acc_l2.extend(cuda.to_cpu(cp.asarray(aux2)))
    else:
        adv_acc_l2 = None
        noise_adv_acc_l2 = None

    print("Exe. time = {}".format(time.perf_counter() - start_time))
    return adv_acc_linf, noise_adv_acc_linf, adv_acc_l2, noise_adv_acc_l2

def other_eval(network, x, target, name_str):
    """ Empirical external perturbation evaluations (L2 and Linf PGD attacks, Gaussian noise perturbation) and SSNR linear corruption)

    Args:
        network: NNAgent object containing the model subject to the pertubation
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

def robustness_eval(network, x, target):
    """ Obtain through numerical integration the variables required to certify the robustness for different input sd

    Args:
        network: NNAgent object containing the model subject to the pertubation
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
    # for sd in np.linspace(0.11, 1.65, 15):
    for sd in np.linspace(0.03, 1.01, 15):
        # for i, sd in enumerate(np.arange(0.41, 2.87, 0.41)):
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
    """ Calculates sample estimate of output metrics (classification probability and smoothed margin)

    Args:
        network: NNAgent object containing the model subject to the pertubation
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

def measurements(network, x, target, dest, robustness = True, num_int = True, sample_est = True):
    """ Calls functions to carry all robustness measurements for the currently loaded NN and stores it into a dictionary that will be used to save the data into .npy files

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
            measurements_dict['p_c'], measurements_dict['p_ru'], measurements_dict['smooth_margin'], measurements_dict['mean_out'], measurements_dict['var_out'] = robustness_eval(network, corr_x, cuda.to_cpu(corr_t.array))
        else:   
            print("Files found for current settings. Skipping the numerical integration of classification probability and smoothed margin.")

    if sample_est:
        if any(not os.path.isfile(name_str + measurement) for measurement in ['emp_p_c.npy', 'emp_margin_mean.npy', 'emp_margin_var.npy']):
            measurements_dict['emp_p_c'], measurements_dict['emp_margin_mean'], measurements_dict['emp_margin_var'] = emp_evals(network, x, target)
        else:
            print("Files found for current settings. Skipping the sample estimate of classification probability and smoothed margin.")

    save_measurements(name_str, measurements_dict)

def list_nets(net_save_dir, loss = None, d = 0, x_var = 0, load_mode = 'load_all'):
    """ Parses trained net files for robustness measurements

    Args:
        net_save_dir: directory containing the trained nets to be loaded
        loss: loss function of trained nets to be loaded
        d: hyperparameter d of the trained nets to be loaded
        x_var: hyperparameter x_var of the trained nets to be loaded
        load_mode: 'load_all' computes robustness measurements for all nets in the folder, anything else loads trained nets with specified loss, d and x_var

    Returns: list of all file names to be loaded

    """
    if load_mode == 'load_all':
        print('### Measurements will be carried for all trained nets')
        files = glob.glob(net_save_dir + "/trained*")
    else:
        print('### Measurements will be carried for {} loss, d = {} and x_var = {}'.format(loss, d, x_var))
        files = glob.glob(net_save_dir + "/trained_loss_{}*_x_var_{}_d_{}_*".format(loss, x_var, d))
    return files

def save_measurements(name_str, measurements_dict):
    """ Saves the robustness measurements data into specified folder

    Args:
        name_str: string containing the destination folder and the part of the file name common to all files
        measurements_dict: dictionary containing the measurement name used as ending of the file name and the data itself saved as .npy file

    """
    print("### Saving collected data into .npy files ###")
    for measure_name, measure in measurements_dict.items():
        if measure is not None:
            np.save(name_str + measure_name, measure)

def manage_gpu():
    pass
