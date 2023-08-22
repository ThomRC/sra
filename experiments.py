#!/usr/bin/env python
# coding: utf-8
from chainer import Variable, initializers
import cupy as cp
import sys
import os
import datetime
import pickle
import io
import measurements.performance.robustness as robustness
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def change_gpu(model, gpu):
    for link in range(len(model)):
        if hasattr(model[link], 'W'):
            model[link].W.to_gpu(gpu)
            model[link].ortho_w.to_gpu(gpu)
        if hasattr(model[link], 'b') and model[link].b is not None:
            model[link].b.to_gpu(gpu)

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        return super(RenameUnpickler, self).find_class(renamed_module, name)

class RenameUnpickler2(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "BNN_mod_bu":
            renamed_module = "BNN_mod_normal"
        return super(RenameUnpickler2, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = renamed_load(fo)
    return dict

def save_net(network, save_dir, net_name, save_data = 1):
    if save_data == 1:
        print("####Saving trained net in the specified directory!")
        network.model.valid_in = network.valid_in[0:2000]
        network.model.valid_out = network.valid_out[0:2000]
        network.model.tr_idx = network.tr_idx
        network.model.te_idx = network.te_idx

        with open(save_dir + '/' + net_name,"wb") as fdata:
            pickle.dump(network.model, fdata)
    else:
        print("####Trained net not saved")

    if save_data == 1:
        print("####Saving data in the specified directory!")
        network.save_data()

    else:
        print("####Not saving any data")

def settings(gpu):
    ############################################################################
    #
    # FIXED PARAMETERS - DON'T TOUCH
    #
    ############################################################################
    save_data = 1 # OV: 1
    ortho_ws = True # OV: True

    ############################################################################
    # DATASET SETTINGS
    dataset = 'MNIST' #'MNIST', 'cifar10' or 'cifar100'

    in_norm = False # OV: False # Whether the training inputs should be normalized or not along all training data
    # Sets the range of each pixel. Ex:
    ### in_interval = 1 and in_center = 0.5 means each pixel in [0,1]
    ### in_interval = 2 and in_center = 0 means each pixel in [-1,1]
    in_interval = 1.  # OV: 1. defines the interval for the input data to be rescaled: from 8 bits ([0,255]) to the new interval ([0, in_interval])
    in_center = 0.5 # OV: 0.5

    tr_size = 60000
    batch_size = 128 # # OV: 128
    # /DATASET SETTINGS
    ############################################################################

    ############################################################################
    # ARCHITECTURE SETTINGS
    units = 512 # # OV: 512 units per layer
    in_padding = True
    # fc_hl_units = [units]
    # fc_hl_units = [units,units]
    fc_hl_units = [units,units,units]
    # fc_hl_units = [units, units, units, units, units, units]
    # IN CASE OF EACH LAYER HAVING DIFFERENT # OF UNITS PLEASE CHANGE IT HERE MANUALLY

    fc_hl = len(fc_hl_units)
    hl_units.extend(fc_hl_units.copy()) # for lists
    nobias = True # OV: True
    initial_bias = None # OV: None
    # /ARCHITECTURE SETTINGS
    ############################################################################

    ############################################################################
    # EXPERIMENTAL SETTINGS
    ############################################################################
    n_exp = 3  # OV = 3 Defines how many trainings are run for each hyperparameter combination
    n_epoch = 300  # OV - 300 number of epochs for each training
    ############################################################################
    # UPDATE ALGORITHM
    ############################################################################
    schedule = 'cos-ann' #lr decay type 'cos-ann' or 'cst'
    adam_beta1 = 0.9 # OV: 0.9
    adam_beta2 = 0.999 # OV: 0.999
    # lr = 0.001  # OV: 0.001 # CIFAR-10
    lr = 0.01 # OV: 0.01 # MNIST
    ############################################################################
    # INITIZALIZATION
    ############################################################################
    init = 'orthogonal' # OV: 'orthogonal' # if the weights should be initialized as 'orthogonal' or 'random'
    batchnorm = False

    if not sys.argv[3]:
        raise RuntimeError('Three arguments required: \n '
                           'loss: loss functions to be used \n'
                           'd: margin enforcement hyperp. \n'
                           'x_var: variance of input Gaussian noise')
    else:
        loss = sys.argv[1]
        d = cp.asarray(float(sys.argv[2]), dtype = cp.float32)
        x_var = cp.asarray(float(sys.argv[3]), dtype = cp.float32)

    if init in {'random', 'orthogonal'}:
        if init == "random":
            initializer = initializers.Orthogonal(scale=1.0)
            bjorck_config = None
        elif init == "orthogonal":
            initializer = initializers.Orthogonal(scale = 1.0)
            # config used for bjorck orthog. on paper exps: 'beta': 0.5,'iter': 30, 'order': 1, 'safe_scaling': True
            bjorck_config = {'beta': 0.5, 'iter': 30, 'order': 1, 'safe_scaling': True, 'dynamic_iter': True}
    else:
        raise RuntimeError('Chosen initialization couldn\'t be recognized\n'
                           'Currently implemented: '
                           '\n- Random (\'random\')'
                           '\n- Orthogonal (\'orthogonal\')')

    kwargs_exp = {'gpu': gpu, 'dataset': dataset, 'tr_size': tr_size, 'normalization': in_norm, 'batchnorm': batchnorm, 'in_padding': in_padding, 'fc_hl': fc_hl, 'hl_units': hl_units, 'in_interval': in_interval, 'in_center': in_center, 'n_exp': n_exp, 'n_epoch': n_epoch, 'save_data': save_data, 'save_net': save_data, 'batch_size': batch_size}
    kwargs_model = {'ortho_ws': ortho_ws, 'loss': loss, 'x_var': x_var, 'd': d, 'init': init, 'pad': pad, 'nobias': nobias, 'initial_w': initializer, 'initial_bias': initial_bias, 'lr': lr, 'beta1': adam_beta1, 'beta2': adam_beta2, 'schedule': schedule, 'bjorck_config': bjorck_config}
    kwargs = {**kwargs_exp, **kwargs_model}
    return kwargs

def experiment_routine(**kwargs):

    # Creates string to describe whether the input is normalized across all data or just rescaled to specified interval
    if kwargs['normalization']:
        in_str = "norm_in"
    else:
        in_str = "in_int_{}_cent_{}".format(kwargs['in_interval'], kwargs['in_center'])

    # Creates string with current architecture as "*number_of_layers*HL_*#unitsHL1*x*#unitsHL1*x...x*#unitsHLn*"
    # to be used when storing the trained nets
    nhl = str(len(kwargs['hl_units']))
    hl_unit_str = nhl + 'HL_'
    for i in range(len(kwargs['hl_units'])):
        hl_unit_str += str(kwargs['hl_units'][i]) + 'x'
    hl_unit_str = hl_unit_str[0:-1]

    date = datetime.datetime.today().strftime('%Y-%m-%d')

    curr_path, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "vol":
        prev_path = curr_path
        curr_path, curr_fold = os.path.split(curr_path)

    print(prev_path)
    if kwargs['bjorck_config']['iter'] == 0:
        dest_dir = prev_path + "/experiments/no_ortho/loss_{}/{}/{}/{}/init_{}/reg_{}/{}/{}_tr/{}/".format(
        kwargs['loss'], kwargs['dataset'], kwargs['arch'], hl_unit_str, kwargs['init'], kwargs['reg'], in_str, kwargs['tr_size'], date)
    else:
        dest_dir = prev_path + "/experiments/loss_{}/{}/{}/{}/init_{}/reg_{}/{}/{}_tr/{}/".format(
        kwargs['loss'], kwargs['dataset'], kwargs['arch'], hl_unit_str, kwargs['init'], kwargs['reg'], in_str, kwargs['tr_size'], date)
    print("Destination folder \n{}".format(dest_dir))
    j = 0
    n_exp = kwargs['n_exp']
    fail = 0
    success = 0
    while success < n_exp:
        # Sets the name of the destination folder for current training with the main training settings
        exp_name = "loss_{}_ep_{}_lr_{}_x_var_{}_d_{}_{}".format(kwargs['loss'], kwargs['n_epoch'], kwargs['lr'], kwargs['x_var'], kwargs['d'], j)
        # Changes the number tag of destination folder until a number that wasn't used
        while os.path.exists(dest_dir + exp_name):
            j += 1
            exp_name = "loss_{}_ep_{}_lr_{}_x_var_{}_d_{}_{}".format(kwargs['loss'], kwargs['n_epoch'], kwargs['lr'],  kwargs['x_var'], kwargs['d'], j)

        save_dir = dest_dir + exp_name
        if 1 in [kwargs['save_data'], kwargs['save_net']]:
            try:
                original_umask = os.umask(0)
                os.makedirs(dest_dir + exp_name, mode=0o770)
            finally:
                os.umask(original_umask)

        kwargs_train = {'exp_count': success + fail, 'success': success, 'save_dir': save_dir}
        kwargs = {**kwargs, **kwargs_train}
        print("#### Training {}".format(success + fail + 1))
        print("## X_var: {} | d: {} | lr: {} | tr_size: {}".format(kwargs['x_var'], kwargs['d'], kwargs['lr'], kwargs['tr_size']))
        tr_result = trainer.run_sim(**kwargs)

        if tr_result == 0:
            fail += 1
        else:
            success += 1

    print("*****************************************************************")
    print("The number of fails = {} and successes = {}".format(fail, success))
    print("*****************************************************************")

    return kwargs

def build_net(**kwargs):
    network = trainer.NNAgent(**kwargs)
    network.prepare_data(**kwargs)
    network.set_model_parameter(**kwargs)
    return network

if __name__ == '__main__':
    gpu = 0
    cp.cuda.Device(gpu).use()
    ### ADD REQUEST FOR USED DIRECTORY
    # IN THAT DIRECTORY A FOLDER WITH DATASET WILL BE CREATED
    # A FOLDER NAMED WITH THE MODEL WILL BE CREATED
    # INSIDE, A FOLDER WITH THE DATE OF THE TRAINING IS CREATED CONTAINING THE TRAINING DATA AND TRAINED MODEL
    while True:
        import trainer
        aux = 'train'
        # aux = 'load'

        while aux not in ['train', 'load']:
            print('The chosen input can\'t be recognized. Please specify it again.')
            aux = input('Do you want to train or load a trained net? \nTrain -> train\nLoad -> load\n')  # 'logn' or 'normal'

        kwargs = settings(gpu)
        if aux == 'train':
            args = experiment_routine(**kwargs)
            sys.exit("Exiting training")
            while True:
                aux = input('Do you want to leave? [y] Yes [n] No\n')
                if aux == 'y':
                    sys.exit("Exiting training")

        elif aux == 'load':
            while True:
                # dest = '/dock/thomas/vol/trained_nets_MNIST/'
                # csvname = 'GPU{}.csv'.format(gpu)
                # csvname = 'Zhen_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_Zhen/'
                # csvname = 'Zhen_GPU{}.csv'.format(gpu)
                # csvname = 'Zhen2_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_PC_lb/'
                # csvname = 'PC_lb_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_Margin/'
                # csvname = 'Margin_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_MC_hinge/'
                # csvname = 'MC_hinge_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_SCE/'
                # csvname = 'SCE_GPU{}.csv'.format(gpu)
                # dest = '/dock/thomas/vol/trained_nets_MNIST_SC-SCE/'
                # csvname = 'SC-SCE_GPU{}.csv'.format(gpu)
                dest = '/dock/thomas/vol/trained_nets_cifar10_SM_lb/'
                csvname = 'SM_lb_GPU{}.csv'.format(gpu)

                filelist = robustness.readcsv(dest, csvname)
                import re

                for i in range(len(filelist)):
                    net_path = dest + filelist[i][0]
                    save_dir, file_name = os.path.split(net_path)
                    print(save_dir)
                    print(file_name)
                    kwargs_load = {'exp_count': -1, 'success': -1, 'save_dir': save_dir}
                    kwargs = {**kwargs, **kwargs_load}
                    network = build_net(**kwargs)
                    network.model = unpickle(net_path)
                    change_gpu(network.model, gpu)
                    network.prepare_data(mode='load', **kwargs)

                    x_m = Variable(network.te_x)
                    target = Variable(network.te_y)
                    network.valid_in = Variable(network.te_x)
                    network.valid_out = Variable(network.te_y)

                    result = re.search('x_var_(.*)_d', net_path)
                    x_var = float(result.group(1))
                    result = re.search('{}_d_(.*)_'.format(x_var), net_path)
                    d = float(result.group(1))

                    network.x_var = x_var
                    network.d = d
                    network.model.intvl_in = network.intvl_in
                    network.model.center_in = network.center_in

                    for j in range(len(network.model)):
                        network.model[j].config['dynamic_iter'] = True
                        network.model[j].dynamic_iter = True

                    robustness.accuracy(network, x_m, target, dest)
                    # robustness.norm_test(network, x_m, dest)
                    robustness.measurements(network, x_m, target, dest)
                    # robustness.ssnr_metrics(network, x_m, target, dest)

                sys.exit("Exiting loading mode")