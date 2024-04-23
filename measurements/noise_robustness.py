import cupy as cp

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
        sd_step = 0.08
    elif network.dataset == "cifar100":
        acc_threshold = 0.013
        sd_step = 0.16

    noise_acc = []
    samples = 100
    sd = 0
    aux = cp.zeros(samples)
    while sd == 0 or cp.mean(aux) > acc_threshold and sd < 10:
        sd += sd_step
        aux = cp.zeros(samples)
        for i in range(samples):
            aux[i] = network.model.validation(x, target, noise_in=True, sd=sd, train=False)

        noise_acc.append(cp.asarray(aux).mean())

    return noise_acc
