import time
import cupy as cp

def linf_pgd(network, x, target):
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

def l2_pgd(network, x, target):
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
