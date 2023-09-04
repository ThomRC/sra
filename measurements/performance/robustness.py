from chainer import cuda, Variable, no_backprop_mode
import cupy as cp
import numpy as np

import os
import glob
import time
# Import the class to build the model and train the net
import stats_func as utils
from multiprocessing import Pool

def linf_attacks(network, x, t):
    if network.dataset == "MNIST":
        radius_step = 0.025
    elif network.dataset == "cifar10":
        radius_step = 0.025
    scale = 1.0
    cw_cut = 0
    loss_func = 'CW'  ## 'crossentropy', 'CW', 'SCW', 'sm1' 'snmse', 'smape'
    num_steps = 200
    step_norm = 'sign'  ## 'sign', 'inf', '2'
    eps_norm = 'inf'  ## 'inf', '2'
    adv_acc = []
    noise_adv_acc = []
    samples = 5
    radius = 0
    aux = cp.zeros(samples)
    aux2 = cp.zeros(samples)
    while radius == 0 or cp.sum(aux) > 0:
        aux = cp.zeros(samples)
        aux2 = cp.zeros(samples)
        radius += radius_step
        for i in range(samples):
            step_size = 0.08 * radius
            t1_1 = time.perf_counter()
            x_adv = network.model.projected_gradient_descent(x, t, num_steps=num_steps,
                                                             step_size=step_size, eps=radius, clamp=(
                    network.center_in - network.intvl_in / 2, network.center_in + network.intvl_in / 2), eps_norm=eps_norm,
                                                             step_norm=step_norm, loss_func=loss_func, pert_scale=scale,
                                                             cw_cut=cw_cut)

            aux[i] = network.model.validation(Variable(x_adv), t, noise_in=False, train=False)
            print("Accuracy for Linf radius = {} is {} #{}".format(radius, aux[i], i))
            aux2[i] = network.model.validation(Variable(x_adv), t, noise_in=True, sd=0.1, train=False)
            print("Noise accuracy for Linf radius = {} is {} #{}".format(radius, aux2[i], i))
            t2_1 = time.perf_counter()
            print("Exe. time = {}".format(t2_1 - t1_1))

        adv_acc.append(cp.asarray(aux).min())
        noise_adv_acc.append(cp.asarray(aux2).min())

    return adv_acc, noise_adv_acc

def l2_attacks(network, x, t):
    if network.dataset == "MNIST":
        radius_step = 0.32
    elif network.dataset == "cifar10":
        radius_step = 0.32
    scale = 1.0
    cw_cut = 0
    loss_func = 'CW'  ## 'crossentropy', 'CW', 'SCW', 'sm1' 'snmse', 'smape'
    num_steps = 200
    step_norm = '2'  ## 'sign', 'inf', '2'
    eps_norm = '2'  ## 'inf', '2'
    adv_acc = []
    noise_adv_acc = []
    samples = 5
    radius = 0
    aux = cp.zeros(samples)
    aux2 = cp.zeros(samples)
    while radius == 0 or cp.sum(aux) > 0:
        radius += radius_step
        aux = cp.zeros(samples)
        aux2 = cp.zeros(samples)
        for i in range(samples):
            step_size = 0.08 * radius
            t1_1 = time.perf_counter()
            x_adv = network.model.projected_gradient_descent(x, t, num_steps=num_steps,
                                                             step_size=step_size, eps=radius, clamp=(
                    network.center_in - network.intvl_in / 2, network.center_in + network.intvl_in / 2), eps_norm=eps_norm,
                                                             step_norm=step_norm, loss_func=loss_func, pert_scale=scale,
                                                             cw_cut=cw_cut)
            aux[i] = network.model.validation(Variable(x_adv), t, noise_in=False, train=False)
            print("Accuracy for L2 radius = {} is {} #{}".format(radius, aux[i], i))
            aux2[i] = network.model.validation(Variable(x_adv), t, noise_in=True, sd=0.1, train=False)
            print("Noise accuracy for L2 radius = {} is {} #{}".format(radius, aux2[i], i))
            t2_1 = time.perf_counter()
            print("Exe. time = {}".format(t2_1 - t1_1))

        adv_acc.append(cp.asarray(aux).min())
        noise_adv_acc.append(cp.asarray(aux2).min())

    return adv_acc, noise_adv_acc

def gaussian_noise_acc(network, x, t):
    # CHECK LATER if theres a more efficient way to compute instead of using the for loop for each sample
    if network.dataset == "MNIST":
        acc_threshold = 0.13
        sd_step = 0.32
    elif network.dataset == "cifar10":
        acc_threshold = 0.13
        sd_step = 0.032
    elif network.dataset == "cifar100":
        acc_threshold = 0.013
        sd_step = 0.032

    noise_acc = []
    samples = 5
    sd = 0
    aux = cp.zeros(samples)
    while sd == 0 or cp.mean(aux) > acc_threshold and sd < 10:
        sd += sd_step
        aux = cp.zeros(samples)
        for i in range(samples):
            aux[i] = network.model.validation(x, t, noise_in=True, sd=sd, train=False)
            print("Noise accuracy for sd = {} is {} #{}".format(sd, aux[i], i))

        noise_acc.append(cp.asarray(aux).mean())

    return noise_acc

def ssnr_eval_acc(network, x, t):
    noise_acc = []
    samples = 10
    for ssnr_ratio in np.linspace(0.95, 0, 20):
        aux = cp.zeros(samples)
        for i in range(samples):
            aux[i] = network.model.validation(x, t, noise_in=True, sd=1.0, train=False, ssnr = True, ssnr_ratio = ssnr_ratio)

        print("Accuracy for SSNR = {} is {}".format(ssnr_ratio, cp.asarray(aux).mean()))
        noise_acc.append(cp.asarray(aux).mean())

    return noise_acc

def perturbation_eval(network, x, t, dest):
    print("### Robustness tests (L2, Linf attacks and Gaussian noise) ###")

    adv_acc_linf = []
    noise_adv_acc_linf = []
    adv_acc_l2 = []
    noise_adv_acc_l2 = []
    noise_acc = []
    ssnr_acc = []

    acc = network.model.validation(x, t)
    acc_noise = network.model.validation(x, t, noise_in=True, sd=0.1, train=False)
    print("Clean accuracy : {}".format(acc))
    print("Clean noise accuracy : {}".format(acc_noise))

    adv_acc_linf.append(acc)
    noise_adv_acc_linf.append(acc_noise)
    adv_acc_l2.append(acc)
    noise_adv_acc_l2.append(acc_noise)
    noise_acc.append(acc)
    ssnr_acc.append(acc)

    if os.path.isfile(name_str + 'SSNR_acc.npy'):
        aux1 = ssnr_eval_acc(network, x, t)
        ssnr_acc.extend(aux1)
    else:
        ssnr_acc = 0
    if os.path.isfile(name_str + 'gauss_noise.npy'):
        aux1 = gaussian_noise_acc(network, x, t)
        noise_acc.extend(aux1)
    else:
        noise_acc = 0
    if os.path.isfile(name_str + 'linf_adv_acc.npy'):
        aux1, aux2 = linf_attacks(network, x, t)
        adv_acc_linf.extend(aux1)
        noise_adv_acc_linf.extend(aux2)
    else:
        adv_acc_linf = 0
        noise_adv_acc_linf = 0
    if os.path.isfile(name_str + 'l2_adv_acc.npy'):
        aux1, aux2 = l2_attacks(network, x, t)
        adv_acc_l2.extend(aux1)
        noise_adv_acc_l2.extend(aux2)
    else:
        adv_acc_l2 = 0
        noise_adv_acc_l2 = 0

    return cuda.to_cpu(cp.asarray(adv_acc_linf)), cuda.to_cpu(cp.asarray(noise_adv_acc_linf)), cuda.to_cpu(cp.asarray(adv_acc_l2)), cuda.to_cpu(cp.asarray(noise_adv_acc_l2)), cuda.to_cpu(cp.asarray(noise_acc)), cuda.to_cpu(cp.asarray(ssnr_acc))

def robustness_eval(network, x, t):
    ## Obtain through numerical integration the variables required to certify the robustness for different input sd
    # 1) Probability of correct classification and ru probability
    # 2) Smoothed margin
    t1 = time.perf_counter()
    print("### Numerical robustness certification metrics (classification probability and smoothed margin) ###")
    # The first element of the lists are for sd = 0
    _, mean_s, var_s, _, _ = network.model.moment_propagation(len(network.model), x, 0, x.array)
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
            _, mean_s, var_s, _, _ = network.model.moment_propagation(len(network.model), x, sd ** 2, x.array)

        mean = cuda.to_cpu(mean_s.array)
        var = cuda.to_cpu(var_s.array)
        mean_out_arr.append(mean)
        var_out_arr.append(var)
        aux = [mean, var, t, 4]
        function_params.append(aux)

    p = Pool(32)
    results_pooled1 = p.map(utils.corr_probabilities,function_params)
    results_pooled2 = p.map(utils.smoothed_margin, function_params)
    aux = np.asarray(results_pooled1)

    c_prob = np.vstack((np.ones((1,aux.shape[2])),aux[:,0,:]))
    ru_prob = np.vstack((np.zeros((1,aux.shape[2])),aux[:,1,:]))
    smooth_margin = np.vstack((margin_zero,np.asarray(results_pooled2)))

    t2 = time.perf_counter()
    print("Exe. time for sd {} = {}".format(sd, t2 - t1))

    return c_prob, ru_prob, smooth_margin, np.asarray(mean_out_arr), np.asarray(var_out_arr)

def emp_evals(network, x, target):
    prob_c_arr = []
    margin_mean_arr = []
    margin_var_arr = []

    print("### Sample estimate of corr. prob. and smooth. margin ###")
    samples_arr = [1, 10, 100, 1000, 10000, 100000, 1000000]
    # samples_arr = [1, 10, 100]
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

    return cuda.to_cpu(cp.asarray(prob_c_arr)), cuda.to_cpu(cp.asarray(margin_mean_arr)), cuda.to_cpu(cp.asarray(margin_var_arr))

def w_linf(network):
    aux = cp.zeros(len(network.model))
    for i in range(len(network.model)):
        aux1 = cp.max(cp.sum(cp.clip(network.model[i].ortho_w.array, a_min = 0), axis = 1))
        aux[i] = aux1

    return  cuda.to_cpu(cp.asarray(aux))

def readcsv(dest, csvname):
    import csv

    file = open(dest + csvname)
    csvreader = csv.reader(file)

    rows = []
    for row in csvreader:
        rows.append(row)
        print(row)
    return rows

def measurements(network, x, t, dest):
    units = network.units[0]
    layers = network.n_hl
    epoch = network.epoch
    training_num = network.training_num
    loss = network.loss
    d = network.d
    x_var = network.x_var
    print("Loss: {}".format(loss))
    print("d: {}".format(d))
    print("X_var: {}".format(x_var))
    name_str = dest + 'loss_{}_ep{}_x_var_{}_d_{}_{}_'.format(loss, epoch, x_var, d, training_num)

    target = cuda.to_cpu(t.array)
    _, mean_s, var_s, mean_h, var_h = network.model.moment_propagation(len(network.model), x, 0, x.array)
    ## Obtain the metrics only for correctly classified samples
    ## For this we need a boolean ndarray of correct samples
    corr_class = cuda.to_cpu(cp.argmax(mean_s.array, axis=1)) # ndarray containing the output indices with maximal value
    corr_idx = corr_class == target # boolean ndarray of size = the number of test samples with True where correctly classified 
    corr_samples = np.arange(10000)[corr_idx]  # ndarray containing the indices of correctly classified samples
    corr_x = x[corr_samples,:].reshape((len(corr_samples),-1)) # ndarray containing the test samples that are correclt classified
    corr_t = t[corr_idx] # ndarray containing the labels of correctly classified inputs

    measurements_dict = {}
    measurements_dict['target'] = cuda.to_cpu(corr_t.array)
    measurements_dict['clean_acc'] = len(corr_t.array)/len(target.array)
    measurements_dict['W_Linf_norm'] = w_linf(network)
    measurements_dict['linf_adv_acc'], measurements_dict['linf_adv_acc_noise'], measurements_dict['l2_adv_acc'], measurements_dict['l2_adv_acc_noise'], measurements_dict['gauss_noise'], measurements_dict['SSNR_acc'] = perturbation_eval(network, corr_x, corr_t)
    if any(not os.path.isfile(name_str + measurement) for measurement in ['p_c', 'p_ru', 'smooth_margin', 'mean_out', 'var_out']):
        measurements_dict['p_c'], measurements_dict['p_ru'], measurements_dict['smooth_margin'], measurements_dict['mean_out'], measurements_dict['var_out'] = robustness_eval(network, corr_x, cuda.to_cpu(corr_t.array))
    if any(not os.path.isfile(name_str + measurement) for measurement in ['emp_p_c', 'emp_margin_mean', 'emp_margin_var']):
        measurements_dict['emp_p_c'], measurements_dict['emp_margin_mean'], measurements_dict['emp_margin_var'] = emp_evals(network, x, t)

    save_measurements(name_str, measurements_dict)

def list_nets(net_save_dir):
    files = glob.glob(net_save_dir + "/trained*")
    return files

def save_measurements(name_str, measurements_dict):
    for measure_name, measure in measurements_dict:
        if measure != 0:
            np.save(name_str + measure_name, measure)

def manage_gpu():
    pass
