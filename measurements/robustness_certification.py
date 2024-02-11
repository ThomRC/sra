import scipy.integrate as integrate
import scipy.special as special
import scipy.stats as stats
import statsmodels.stats.proportion as prop
from chainer import no_backprop_mode, cuda
from chainer.functions import relu
import numpy as np

np.random.seed(0)
import code

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
                    # h = F.relu(network[link](h))
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
            # code.interact(banner="Inside unit_sampling:", local=locals())
        return margin_sum/n, (margin_sq_sum/n - (margin_sum/n)**2) * n/(n-1)

def sampleundernoise(network, x, sd, n, dist = 'Normal', radius = 0.3):
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    parallel_samples = 5
    x = network.model.xp.repeat(x.array, parallel_samples, 0) #repeat each entry in 0-dim 
    with no_backprop_mode():
        count_mat = network.model.xp.zeros((10000,10))
        for _ in range(int(n/parallel_samples)):
            h = x + network.model.noise_inj(x, sd, dist, radius)
            for link in range(len(network.model) - 1):
                if hasattr(network.model[link], 'W'):
                    h = relu(network.model[link](h))
                else:
                    h = network.model[link](h)
            h = network.model[link + 1](h)
            # code.interact(banner="Inside unit_sampling:", local=locals())
            corr_out_idx = network.model.xp.sum(network.model.xp.argmax(h.array, axis = 1).reshape((10000,parallel_samples)), axis = 0)
            # corr_out_idx = corr_out == val_out.array
            # count_mat[np.arange(10000),corr_out_idx] += ones_vec[corr_out_idx]
            for i in range(parallel_samples):
                count_mat[np.arange(10000),corr_out_idx[:,i]] += 1
        return count_mat

def sampleundernoise(network, x, sd, n, dist = 'Normal', radius = 0.3):
    # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
    n = 50000
    count_vec = network.model.xp.zeros(10)
    with no_backprop_mode():
        x = network.model.xp.repeat(cp.expand_dims(x.array,0), n, 0)
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

def certify(network, val_in, sd, n, alpha):
    # count_mat0 = cuda.to_cpu(sampleundernoise(network,val_in, sd, n0))
    # cA = np.argmax(count_mat0,axis = 1)
    count_mat = cuda.to_cpu(sampleundernoise(network,val_in, sd, n))
    print("Finished sampling")
    p = np.zeros(10000)
    radius = np.zeros(10000)
    cA = np.argmax(count_mat,axis = 1)
    nA = count_mat[np.arange(10000),cA]
    normal = stats.norm
    for i in range(10000):
        p[i] = prop.proportion_confint(nA[i], n, 1 - alpha)[0]
        if p[i] > 0.5:
            radius[i] = sd*normal.ppf(p[i])
        else:
            cA[i] = -1
    return cA, radius

def lipconstant(network):
    lip_layer = 1
    for link in range(len(network) - 1):
        lip_layer *= spectralradius(network[link].W.array) if hasattr(network[link], 'W')