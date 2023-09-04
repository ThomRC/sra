#!/usr/bin/env python
# coding: utf-8
from chainer import initializers
import cupy as cp
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def settings():
    """ Function to return dictionary with all experimental and training settings

    Args:
        gpu: number of GPU to be used

    Returns: kwargs dictionary containing all experimental settings

    """
    ############################################################################
    #
    # FIXED PARAMETERS - DON'T TOUCH
    #
    ############################################################################
    save_data = True  # Original Value (OV): True # True saves data collected during training like loss, training error, test error, etc.
    save_net = True  # Original Value (OV): True # True saves trained net
    ortho_ws = True  # OV: True

    ############################################################################
    # DATASET SETTINGS
    dataset = 'MNIST'  # 'MNIST', 'cifar10' or 'cifar100'

    in_norm = False  # OV: False # Whether the training inputs should be normalized or not along all training data
    # Sets the range of each pixel. Ex:
    ### in_interval = 1 and in_center = 0.5 means each pixel in [0,1]
    ### in_interval = 2 and in_center = 0 means each pixel in [-1,1]
    in_interval = 1.  # OV: 1. defines the interval for the input data to be rescaled: from 8 bits ([0,255]) to the new interval ([0, in_interval])
    in_center = 0.5  # OV: 0.5

    tr_size = 60000
    batch_size = 128  # # OV: 128
    # /DATASET SETTINGS
    ############################################################################

    ############################################################################
    # ARCHITECTURE SETTINGS
    units = 512  # # OV: 512 units per layer
    in_padding = True
    # fc_hl_units = [units]
    # fc_hl_units = [units,units]
    fc_hl_units = [units, units, units]
    # fc_hl_units = [units, units, units, units, units, units]
    # IN CASE OF EACH LAYER HAVING DIFFERENT # OF UNITS PLEASE CHANGE IT HERE MANUALLY

    fc_hl = len(fc_hl_units)
    hl_units.extend(fc_hl_units.copy())  # for lists
    nobias = True  # OV: True
    initial_bias = None  # OV: None

    # Creates string with current architecture as "*number_of_layers*HL_*#unitsHL1*x*#unitsHL1*x...x*#unitsHLn*"
    # to be used when storing the trained nets
    nhl = str(len(hl_units))
    arch = nhl + 'HL_'
    for i in range(len(hl_units)):
        arch += str(hl_units[i]) + 'x'
    arch = arch[0:-1]
    # /ARCHITECTURE SETTINGS
    ############################################################################
    curr_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    print(curr_dir)
    print(curr_fold)
    net_save_dir = curr_dir + "/trained_NNs/{}/".format(dataset) + "{}".format(arch)
    if save_net:
        try:
            original_umask = os.umask(0)
            os.makedirs(net_save_dir, mode=0o770)
        finally:
            os.umask(original_umask)

    ############################################################################
    # EXPERIMENTAL SETTINGS
    ############################################################################
    n_exp = 3  # OV = 3 Defines how many trainings are run for each hyperparameter combination
    n_epoch = 300  # OV - 300 number of epochs for each training
    ############################################################################
    # UPDATE ALGORITHM
    ############################################################################
    schedule = 'cos-ann'  # lr decay type 'cos-ann' or 'cst'
    adam_beta1 = 0.9  # OV: 0.9
    adam_beta2 = 0.999  # OV: 0.999
    # lr = 0.001  # OV: 0.001 # CIFAR-10
    lr = 0.01  # OV: 0.01 # MNIST
    ############################################################################
    # INITIZALIZATION
    ############################################################################
    init = 'orthogonal'  # OV: 'orthogonal' # if the weights should be initialized as 'orthogonal' or 'random'
    batchnorm = False

    if not sys.argv[5]:
        raise RuntimeError('Four arguments required: \n'
                           'loss: loss functions to be used \n'
                           'd: margin enforcement hyperparameter \n'
                           'x_var: variance of input Gaussian noise \n'
                           'Mode: \'train\' or \'load\' \n'
                           'gpu: number of gpu to be used')
    else:
        loss = sys.argv[1]
        d = cp.asarray(float(sys.argv[2]), dtype=cp.float32)
        x_var = cp.asarray(float(sys.argv[3]), dtype=cp.float32)
        mode = sys.argv[4]

    if init in {'random', 'orthogonal'}:
        if init == "random":
            initializer = initializers.Orthogonal(scale=1.0)
            bjorck_config = None
        elif init == "orthogonal":
            initializer = initializers.Orthogonal(scale=1.0)
            # config used for bjorck orthog. on paper exps: 'beta': 0.5,'iter': 30, 'order': 1, 'safe_scaling': True
            bjorck_config = {'beta': 0.5, 'iter': 30, 'order': 1, 'safe_scaling': True, 'dynamic_iter': True}
    else:
        raise RuntimeError('Chosen initialization couldn\'t be recognized\n'
                           'Currently implemented: '
                           '\n- Random (\'random\')'
                           '\n- Orthogonal (\'orthogonal\')')

    kwargs_exp = {'gpu': gpu, 'dataset': dataset, 'tr_size': tr_size, 'normalization': in_norm, 'batchnorm': batchnorm,
                  'in_padding': in_padding, 'fc_hl': fc_hl, 'hl_units': hl_units, 'in_interval': in_interval,
                  'in_center': in_center, 'n_exp': n_exp, 'n_epoch': n_epoch, 'save_data': save_data,
                  'save_net': save_net, 'net_save_dir': net_save_dir, 'batch_size': batch_size}
    kwargs_model = {'arch': arch, 'ortho_ws': ortho_ws, 'loss': loss, 'x_var': x_var, 'd': d, 'init': init, 'pad': pad,
                    'nobias': nobias, 'initial_w': initializer, 'initial_bias': initial_bias, 'lr': lr,
                    'beta1': adam_beta1, 'beta2': adam_beta2, 'schedule': schedule, 'bjorck_config': bjorck_config}
    kwargs = {**kwargs_exp, **kwargs_model}
    return mode, kwargs
