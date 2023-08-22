from chainer import cuda, Variable, no_backprop_mode
import cupy as cp
import numpy as np

import os
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
    if network.dataset == "MNIST":
        sd_step = 0.32
    elif network.dataset == "cifar10":
        sd_step = 0.032

    noise_acc = []
    samples = 5
    sd = 0
    aux = cp.zeros(samples)
    while sd == 0 or cp.mean(aux) > 0.11 and sd < 10:
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

def perturbation_eval(network, x, t):
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

    aux1 = ssnr_eval_acc(network, x, t)
    ssnr_acc.extend(aux1)
    aux1 = gaussian_noise_acc(network, x, t)
    noise_acc.extend(aux1)
    aux1, aux2 = linf_attacks(network, x, t)
    adv_acc_linf.extend(aux1)
    noise_adv_acc_linf.extend(aux2)
    aux1, aux2 = l2_attacks(network, x, t)
    adv_acc_l2.extend(aux1)
    noise_adv_acc_l2.extend(aux2)

    return adv_acc_linf, noise_adv_acc_linf, adv_acc_l2, noise_adv_acc_l2, noise_acc, ssnr_acc

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
    print("Exe. time for sd {}= {}".format(sd, t2 - t1))

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

def ssnr_metrics(network, x, t, dest):
    units = network.units[0]
    layers = network.n_hl
    epoch = network.epoch
    n_sim = network.n_sim
    loss = network.loss
    d = network.d
    x_var = network.x_var
    print("Loss: {}".format(loss))
    print("d: {}".format(d))
    print("X_var: {}".format(x_var))

    target = cuda.to_cpu(t.array)
    _, mean_s, var_s, mean_h, var_h = network.model.moment_propagation(len(network.model), x, 0, x.array)
    ## Obtain the metrics only for correctly classified samples
    ## For this we need a np boolean array of correct samples
    corr_class = cuda.to_cpu(cp.argmax(mean_s.array, axis=1))
    corr_idx = corr_class == target
    corr_samples = np.arange(10000)[corr_idx]
    corr_x = x[corr_samples,:].reshape((len(corr_samples),-1))
    corr_t = t[corr_idx]

    aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_SSNR_acc.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)
    while os.path.exists(dest + aux):
        n_sim += 1
        aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_SSNR_acc.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)

    print("### SSNR robustness ###")
    ssnr_acc = [cp.asarray(1)]

    aux1 = ssnr_eval_acc(network, corr_x, corr_t)
    ssnr_acc.extend(aux1)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_SSNR_acc'.format(layers, units, epoch, loss, d, x_var, n_sim), cuda.to_cpu(cp.asarray(ssnr_acc)))

def accuracy(network, x, t, dest):
    units = network.units[0]
    layers = network.n_hl
    epoch = network.epoch
    n_sim = network.n_sim
    loss = network.loss
    d = network.d
    x_var = network.x_var

    aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_clean_acc.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)
    while os.path.exists(dest + aux):
        n_sim += 1
        aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_clean_acc.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)

    print("# of trained net: {}".format(n_sim))
    print("Loss: {}".format(loss))
    print("d: {}".format(d))
    print("X_var: {}".format(x_var))
    acc = network.model.validation(x, t)
    acc_noise = network.model.validation(x, t, noise_in=True, sd=0.1, train=False)
    print("Clean accuracy : {}".format(acc))
    print("Clean noise accuracy : {}".format(acc_noise))

    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_clean_acc'.format(layers, units, epoch, loss, d, x_var, n_sim), cuda.to_cpu(cp.asarray(acc)))

def w_linf(network):
    aux = cp.zeros(len(network.model))
    for i in range(len(network.model)):
        aux1 = cp.max(cp.sum(cp.clip(network.model[i].ortho_w.array, a_min = 0), axis = 1))
        aux[i] = aux1

    return  aux

def readcsv(dest, csvname):
    import csv

    file = open(dest + csvname)
    csvreader = csv.reader(file)

    rows = []
    for row in csvreader:
        rows.append(row)
        print(row)
    return rows

def norm_test(network, x_m, dest):
    units = network.units[0]
    layers = network.n_hl
    epoch = network.epoch
    n_sim = network.n_sim
    loss = network.loss
    d = network.d
    x_var = network.x_var
    print("Loss: {}".format(loss))
    print("d: {}".format(d))
    print("X_var: {}".format(x_var))

    aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_norm_test.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)
    while os.path.exists(dest + aux):
        n_sim += 1
        aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_norm_test.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)

    import cupy as cp
    from chainer import cuda
    import scipy.stats as stats
    x_var_arr = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 1.75]
    app_norm_perc = []
    for x_var in x_var_arr:
        outs = cp.asarray(network.model.out_sampling(x_m, cp.sqrt(x_var), samples=50))
        _, mean_s, var_s, mean_h, var_h = network.model.moment_propagation(len(network.model), x_m, x_var, x_m.array)
        std_outs = cuda.to_cpu((outs.T - mean_s.array.reshape((10000, 10, 1))) / cp.sqrt(var_s.array.reshape((10000, 10, 1))))
        ad_stat = []
        rand_idx = np.random.choice(10000, 1000, replace=False)
        for j in rand_idx:
            for i in range(10):
                aux1 = stats.anderson(std_outs[j, i, :])
                ad_stat.append(aux1[0])

        aux = np.asarray(ad_stat)
        app_norm_perc.append(aux[aux <= 2.492].shape[0] / 100)

    print(app_norm_perc)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_norm_test'.format(layers, units, epoch, loss, d, x_var, n_sim), np.asarray(app_norm_perc))


def measurements(network, x, t, dest):
    units = network.units[0]
    layers = network.n_hl
    epoch = network.epoch
    n_sim = network.n_sim
    loss = network.loss
    d = network.d
    x_var = network.x_var
    print("Loss: {}".format(loss))
    print("d: {}".format(d))
    print("X_var: {}".format(x_var))

    target = cuda.to_cpu(t.array)
    _, mean_s, var_s, mean_h, var_h = network.model.moment_propagation(len(network.model), x, 0, x.array)
    ## Obtain the metrics only for correctly classified samples
    ## For this we need a np boolean array of correct samples
    corr_class = cuda.to_cpu(cp.argmax(mean_s.array, axis=1))
    corr_idx = corr_class == target
    corr_samples = np.arange(10000)[corr_idx]
    corr_x = x[corr_samples,:].reshape((len(corr_samples),-1))
    corr_t = t[corr_idx]

    aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_W_Linf_norm.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)
    while os.path.exists(dest + aux):
        n_sim += 1
        aux = 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_W_Linf_norm.npy'.format(layers, units, epoch, loss, d, x_var, n_sim)

    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_target'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(corr_t.array))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_W_Linf_norm'.format(layers, units, epoch, loss, d, x_var, n_sim), cuda.to_cpu(cp.asarray(w_linf(network))))

    adv_acc_linf, noise_adv_acc_linf, adv_acc_l2, noise_adv_acc_l2, noise_acc, ssnr_acc = perturbation_eval(network, corr_x, corr_t)

    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_linf_adv_acc'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(cp.asarray(adv_acc_linf)))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_linf_adv_acc_noise'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(cp.asarray(noise_adv_acc_linf)))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_l2_adv_acc'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(cp.asarray(adv_acc_l2)))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_l2_adv_acc_noise'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(cp.asarray(noise_adv_acc_l2)))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_gauss_noise'.format(layers, units, epoch, loss, d, x_var, n_sim),cuda.to_cpu(cp.asarray(noise_acc)))
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_SSNR_acc'.format(layers, units, epoch, loss, d, x_var, n_sim), cuda.to_cpu(cp.asarray(ssnr_acc)))

    p_c_arr, p_ru_arr, smooth_margin_arr, mean_out_arr, var_out_arr = robustness_eval(network, corr_x, cuda.to_cpu(corr_t.array))

    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_p_c'.format(layers, units, epoch, loss, d, x_var, n_sim), p_c_arr)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_p_ru'.format(layers, units, epoch, loss, d, x_var, n_sim), p_ru_arr)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_smooth_margin'.format(layers, units, epoch, loss, d, x_var, n_sim), smooth_margin_arr)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_mean_out'.format(layers, units, epoch, loss, d, x_var, n_sim), mean_out_arr)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_var_out'.format(layers, units, epoch, loss, d, x_var, n_sim), var_out_arr)

    emp_p_c, emp_margin_mean, emp_margin_var = emp_evals(network, x, t)

    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_emp_p_c'.format(layers, units, epoch), emp_p_c)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_emp_margin_mean'.format(layers, units, epoch), emp_margin_mean)
    np.save(dest + 'HL{}_{}_ep{}_{}_d_{}_x_var_{}_{}_emp_margin_var'.format(layers, units, epoch), emp_margin_var)
