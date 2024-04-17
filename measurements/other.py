import cupy as cp
import numpy as np

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
