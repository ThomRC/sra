import scipy.stats as stats
import statsmodels.stats.proportion as prop
from chainer import no_backprop_mode, cuda
from chainer.functions import relu
import numpy as np
import cupy as cp

def sample_mean_margin(network, val_in, target, sd, n, dist = 'Normal', radius = 0.3):
    """
    Computes the sample mean of the margin of a neural network output

    Args:
        network: The neural network model.
        val_in: The input data.
        target: The target labels.
        sd: The standard deviation of noise injection.
        n: The number of iterations.
        dist: The distribution of noise (default is 'Normal', other is 'Uniform').
        radius: The radius for noise injection in case of 'Uniform' (default is 0.3).

    Returns:
        Tuple of mean margin and margin variance estimate.
    """
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    with no_backprop_mode():
        margin_sum = network.model.xp.zeros(10000)
        margin_sq_sum = network.model.xp.zeros(10000)
        for _ in range(n):
            noise = network.model.noise_inj(val_in, sd, dist, radius)
            h = val_in + noise
            for link in range(len(network.model) - 1):
                if hasattr(network.model[link], 'W'):
                    h = relu(network.model[link](h))
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

def sampleundernoise(network, x, sd, times, dist = 'Normal', radius = 0.3):
    """
    Computes the count vector of classification under noise for a neural network output.
    
    Algorithm from:
    Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. "Certified Adversarial Robustness via Randomized Smoothing "
    Proceedings of the 36th International Conference on Machine Learning (2019)
    
    Args:
        network: The neural network model.
        x: The input data.
        sd: The standard deviation of noise injection.
        times: The number of iterations.
        dist: The distribution of noise (default is 'Normal', other is 'Uniform').
        radius: The radius for noise injection in case of 'Uniform' (default is 0.3).

    Returns:
        Vector representing the count of activations under noise.
    """    
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    n = 50000
    count_vec = network.model.xp.zeros(10)
    with no_backprop_mode():
        x = network.model.xp.repeat(cp.expand_dims(x.array,0), n, 0)
        for _ in range(times):
            noise = network.model.noise_inj(x, sd, dist, radius)
            h = x + noise
            for link in range(len(network.model) - 1):
                if hasattr(network.model[link], 'W'):
                    h = relu(network.model[link](h))
                else:
                    h = network.model[link](h)
            h = network.model[link + 1](h)
            shift_out = h.array - h.array.min()
            max_out = network.model.xp.max(shift_out, axis = 1).reshape((n,1))        
            aux = (shift_out/max_out).astype(dtype=cp.int8)
            count_vec += network.model.xp.sum(aux, axis = 0)
    return count_vec

def predict(network, val_in, sd, n, alpha):
    """
    Predicts the class labels based on count vector statistics and a significance level.
    
    Algorithm from:
    Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. "Certified Adversarial Robustness via Randomized Smoothing "
    Proceedings of the 36th International Conference on Machine Learning (2019)
    
    Args:
        network: The neural network model.
        val_in: The input data.
        sd: The standard deviation of noise injection.
        n: The number of iterations for noise sampling.
        alpha: The significance level for binomial test.

    Returns:
        Predicted class labels based on statistical tests.    
    """    
    count_mat = cuda.to_cpu(sampleundernoise(network,val_in, sd, n))
    sort_count = np.argsort(count_mat,axis = 1)
    cA = sort_count[:,-1]
    cB = sort_count[:,-2]
    nA = count_mat[np.arange(10000),cA]
    nB = count_mat[np.arange(10000),cB]
    p = np.zeros(10000)
    for i in range(10000):
        p[i] = stats.binom_test(nA[i], nA[i] + nB[i], 0.5)
    abst_idx = p > alpha
    cA[abst_idx] = -1
    return cA

def certify(network, x_in, sd, times, alpha):
    """
    Certifies the Randomized Smoothing robustness of a neural network.

    Algorithm from:
    Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. "Certified Adversarial Robustness via Randomized Smoothing "
    Proceedings of the 36th International Conference on Machine Learning (2019)

    Args:
        network: The neural network model.
        x_in: The input data.
        sd: The standard deviation of noise injection.
        times: The number of noise injection iterations.
        alpha: The significance level for confidence interval.

    Returns:
        Tuple of certified class label and robustness radius.    
    """
    count_mat = cuda.to_cpu(sampleundernoise(network, x_in, sd, times))
    cA = np.argmax(count_mat)
    nA = count_mat[cA]
    normal = stats.norm
    p = prop.proportion_confint(nA, times*50000, alpha = 2*alpha, method = "beta")[0]
    if p > 0.5:
        radius = sd*normal.ppf(p)
    else:
        cA = -1
        radius = -1
    return cA, radius

def cohen_cr(network, x, target):
    """
    Measures the randomized smoothing certified radius for given input noise std (used value of 0.125)
    
    Certified radius formula from:
    Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter. "Certified Adversarial Robustness via Randomized Smoothing "
    Proceedings of the 36th International Conference on Machine Learning (2019)
    
    Args:
        network: NNAgent object containing the model subject to the pertubation
        x: input that will be perturbed
        target: correct class array

    Returns: accuracy under noise for each input variance
    """
    out_samples = cuda.to_cpu(network.model.output_sampling(x, 0, 1))
    class_out = np.argmax(out_samples, axis = 1)
    tgt = cuda.to_cpu(target.array)
    corr_idx = np.arange(10000)[class_out==tgt]
    cr_arr = []
    count = 0
    sd = 0.125
    for i in corr_idx:
        print(count)
        count +=1

        cA, radius = certify(network, x[i], sd, 20, 0.001)
        if cA == tgt[i]:
            cr_arr.append(radius)
        else:
            cr_arr.append(-1)
        
    return np.asarray(cr_arr)

def convsv(kernel, input_shape): 
    """
    Computes the singular values of a convolutional layer
    
    Function extracted from:
    Hanie Sedghi, Vineet Gupta, Philip M. Long. "The singular values of convolutional layers"
    International Conference on Learning Representations, 2019.
    """
    kernel = np.moveaxis(kernel, [0, 1, 2, 3], [2, 3, 1, 0])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1]) 
    return np.linalg.svd(transforms, compute_uv=False)

def lipconstant(network):
    """
    Calculates the Lipschitz constant of a neural network.

    Args:
        network: The neural network model.

    Returns:
        Lipschitz constant of the neural network.
    """    
    lip_net = 1
    for link in range(len(network.model)):
        if hasattr(network.model[link], 'W'):
            ortho_w = cuda.to_cpu(network.model[link].ortho_w.array)
            max_shape = np.asarray(ortho_w.shape).max()
            if hasattr(network.model[link], "kernel_size"):
                spectrad = convsv(ortho_w,(max_shape,max_shape)).max()
                print(spectrad)
            else:
                spectrad = np.linalg.svd(ortho_w)[1].max()
                print(spectrad)
            lip_net *= spectrad

    return lip_net

def lip_cr(network, x, target):
    """
    Calculates the Lipschitz-margin certified radius (CR) for a given network and input.

    This function computes the CR based on the Lipschitz constant of the network, the model's logits for the input, and the target class.

    Args:
        network: The neural network model.
        x: The input data.
        target: The target class for which the CR is calculated.

    Returns:
        cr: The certified radius for the given network, input, and target class.
    """    
    lip_cst = lipconstant(network)
    logits = network.model(x, train = False)
    idx_aux = cp.arange(len(target))
    aux2 = cp.zeros_like(logits.array, dtype = 'bool')
    aux2[idx_aux,target.array] = True
    aux3 = cp.logical_not(aux2)
    notc_out = logits[aux3].reshape((len(target),-1))
    not_tgt_max_idx = notc_out.array.argmax(axis=1)
    cr = cuda.to_cpu(logits[idx_aux, target.array].array - notc_out[idx_aux, not_tgt_max_idx].array)/(np.sqrt(2)*lip_cst)
    return cr
