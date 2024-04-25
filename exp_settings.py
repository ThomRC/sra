#!/usr/bin/env python
# coding: utf-8
import os
from chainer import initializers

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # In case the GPU order in the bus is different from the one listed by CUDA

def settings(input_params):
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
    # dataset = 'MNIST'  # 'MNIST', 'cifar10' or 'cifar100'
    dataset = 'cifar10'
    
    augmentation = True
    in_norm = False  # OV: False # Whether the training inputs should be normalized or not along all training data
    # Sets the range of each pixel. Ex:
    ### in_interval = 1 and in_center = 0.5 means each pixel in [0,1]
    ### in_interval = 2 and in_center = 0 means each pixel in [-1,1]
    in_interval = 1.  # OV: 1. defines the interval for the input data to be rescaled: from 8 bits ([0,255]) to the new interval ([0, in_interval])
    in_center = 0.5  # OV: 0.5

    if dataset == 'MNIST':
        tr_size = 60000
    else:
        tr_size = 50000

    batch_size = 128  # # OV: 128
    # /DATASET SETTINGS
    ############################################################################

    ############################################################################
    # STARTS ARCHITECTURE SETTINGS
    arch = 'cnn' # 'mlp' or 'cnn'
    # arch = 'mlp' # 'mlp' or 'cnn'

    units = 512  # # OV: 512 units per layer
    in_padding = False # concatenates channels to match the dimension of lower to higher dimensional transformations
    
    # IN CASE OF EACH LAYER HAVING DIFFERENT # OF UNITS PLEASE CHANGE IT LIKE
    # fc_hl_units = [units1, units2, units3, ...]
    # E.g.: fc_hl_units = [500, 400, 300, 200, 100]    
    if arch == 'cnn':
        cv_hl_kernels = [8, 8, 16, 16]
        cv_hl_ksizes = [4, 3, 4, 3]
        cv_hl_strides = [2, 1, 2, 1]
        fc_hl_units = [units]
    elif arch == 'mlp':
        cv_hl_kernels = []
        cv_hl_ksizes = []
        cv_hl_strides = []  
        fc_hl_units = [units, units, units]

    cv_config = {'channel': cv_hl_kernels, 'ksize': cv_hl_ksizes, 'stride': cv_hl_strides}

    bias = False  # OV: False
    initial_bias = None  # OV: None

    # Creates string with current architecture as "*number_of_layers*HL_*#unitsHL1*x*#unitsHL1*x...x*#unitsHLn*"
    # to be used when storing the trained nets
    cv_hl = len(cv_hl_kernels)
    fc_hl = len(fc_hl_units)
    arch_str = ""

    if arch == "cnn":
        arch_str += str(cv_hl) + 'CVHL_'
        for i in range(cv_hl):
            arch_str += str(cv_hl_kernels[i]) + 'c' + str(cv_hl_ksizes[i]) + 'k' + str(cv_hl_strides[i]) + 's' + 'x'
        arch_str += '_'
    else:
        arch_str = ""
    
    arch_str += str(fc_hl) + 'FCHL_'
    for units in fc_hl_units:
        arch_str += str(units) + 'x'

    arch_str = arch_str[0:-1]
    print("Architecture used: " + arch_str)
    
    # ENDS ARCHITECTURE SETTINGS
    #########################ß###################################################
    curr_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))

    net_save_dir = curr_dir + "/trained_NNs/{}/{}/".format(dataset, arch) + "{}".format(arch_str)
    if save_net:
        if not os.path.isdir(net_save_dir):
            os.makedirs(net_save_dir)

    ############################################################################
    # EXPERIMENTAL SETTINGS
    ############################################################################
    n_exp = 3  # OV = 3 Defines how many trainings are run for each hyperparameter combination
    n_epoch = 300  # OV - 300 number of epochs for each training
    ############################################################################
    # UPDATE ALGORITHM
    ############################################################################
    schedule = 'cos-ann'  # lr decay type 'cos-ann', 'cst' or 'step'
    adam_beta1 = 0.9  # OV: 0.9
    adam_beta2 = 0.999  # OV: 0.999
    # lr = 0.001  # OV: 0.001 # CIFAR-10
    # lr = 0.005  # OV: 0.001 # CIFAR-10
    lr = 0.01  # OV: 0.01 # MNIST
    ############################################################################
    # INITIZALIZATION
    ############################################################################
    init = 'orthogonal'  # OV: 'orthogonal' # if the weights should be initialized as 'orthogonal' or 'random'

    if init in {'random', 'orthogonal'}:
        if init == "random":
            initializer = initializers.Orthogonal(scale=1.0)
            bjorck_config = None
        elif init == "orthogonal":
            initializer = initializers.Orthogonal(scale=1.0)
            # config used for bjorck orthog. on paper exps: 'beta': 0.5,'iter': 30, 'order': 1, 'safe_scaling': True
            bjorck_config = {'beta': 0.5, 'iter': 10, 'order': 1, 'safe_scaling': True, 'dynamic_iter': True}
    else:
        raise RuntimeError('Chosen initialization couldn\'target be recognized\n'
                           'Currently implemented: '
                           '\n- Random (\'random\')'
                           '\n- Orthogonal (\'orthogonal\')')

    kwargs_exp = {'dataset': dataset, 'in_interval': in_interval, 'in_center': in_center, 
                  'tr_size': tr_size, 'batch_size': batch_size, 'normalization': in_norm, 'in_padding': in_padding, 
                  'lr': lr, 'beta1': adam_beta1, 'beta2': adam_beta2, 
                  'schedule': schedule, 'n_exp': n_exp, 'n_epoch': n_epoch, 'save_data': save_data,
                  'save_net': save_net, 'net_save_dir': net_save_dir}
    
    kwargs_model = {'arch': arch, 'arch_str': arch_str, 'cv_hl': cv_hl, 'cv_config': cv_config, 
                    'fc_hl': fc_hl, 'fc_hl_units': fc_hl_units, 'ortho_ws': ortho_ws, 
                    'init': init, 'bias': bias, 'initial_w': initializer, 
                    'initial_bias': initial_bias, 'bjorck_config': bjorck_config}
    kwargs = {**kwargs_exp, **kwargs_model, **input_params}
    return kwargs
