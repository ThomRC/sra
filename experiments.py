#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import datetime
import pickle
import io
import re
import code

from chainer import Variable, initializers
import cupy as cp

from exp_settings import settings
import trainers.trainer as trainer
import measurements.robustness as robustness
from measurements.robustness.utils import list_nets

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # In case the GPU order in the bus is different from the one listed by CUDA

def change_gpu(model, gpu):
    """ Used in case the saved net was in a different GPU from the one used when loading
    Args:
        model: NNAgent object
        gpu: number of the GPU to be transfered to

    """
    for link in range(len(model)):
        if hasattr(model[link], 'W'):
            model[link].W.to_gpu(gpu)
            model[link].ortho_w.to_gpu(gpu)
            if hasattr(model[link], "kernel_size"):
                model[link].mask.to_gpu(gpu)
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

def experiment_routine(**kwargs):
    """ Function to be called to carry multiple trainings under the same settings

    Args:
        **kwargs:

    Returns: kwargs dictionary containing the number of successful and failed trainings

    """
    # Creates string to describe whether the input is normalized across all data or just rescaled to specified interval
    if kwargs['normalization']:
        in_str = "norm_in"
    else:
        in_str = "in_int_{}_cent_{}".format(kwargs['in_interval'], kwargs['in_center'])

    arch = kwargs['arch']
    date = datetime.datetime.today().strftime('%Y-%m-%d')

    curr_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))

    if kwargs['bjorck_config']['iter'] == 0:
        dest_dir = curr_dir + "/trainings/{}/{}/no_ortho/loss_{}/init_{}/{}/{}_tr/{}/".format(
        kwargs['dataset'], arch, kwargs['loss'], kwargs['init'], in_str, kwargs['tr_size'], date)
    else:
        dest_dir = curr_dir + "/trainings/{}/{}/ortho/loss_{}/init_{}/{}/{}_tr/{}/".format(
        kwargs['dataset'], arch, kwargs['loss'], kwargs['init'], in_str, kwargs['tr_size'], date)
    print("Destination folder \n{}".format(dest_dir))

    j = 0
    n_exp = kwargs['n_exp']
    fail = 0
    success = 0
    while success < n_exp:
        # Sets the name of the destination folder for current training with the main training settings
        exp_name = "loss_{}_ep_{}_lr_{}_x_var_{}_d_{}_{}".format(kwargs['loss'], kwargs['n_epoch'], kwargs['lr'], kwargs['x_var'], kwargs['d'], j)
        # Changes the number tag of destination folder until a number that wasn'target used
        while os.path.exists(dest_dir + exp_name):
            j += 1
            exp_name = "loss_{}_ep_{}_lr_{}_x_var_{}_d_{}_{}".format(kwargs['loss'], kwargs['n_epoch'], kwargs['lr'],  kwargs['x_var'], kwargs['d'], j)

        data_save_dir = dest_dir + exp_name
        os.makedirs(data_save_dir)

        kwargs_train = {'exp_count': success + fail, 'success': success, 'data_save_dir': data_save_dir}
        kwargs = {**kwargs, **kwargs_train}
        print("#### Training {}".format(success + fail + 1))
        print("## X_var: {} | d: {} | lr: {} | tr_size: {}".format(kwargs['x_var'], kwargs['d'], kwargs['lr'], kwargs['tr_size']))
        tr_result = trainer.run_training(**kwargs)

        if tr_result == 0:
            fail += 1
        else:
            success += 1

    print("*****************************************************************")
    print("The number of fails = {} and successes = {}".format(fail, success))
    print("*****************************************************************")

    return kwargs

def build_net(**kwargs):
    """Builds a base agent that will be used to load the trained NN
    Make sure that all the architecture settings match
    """
    network = trainer.NNAgent(**kwargs)
    network.prepare_data(**kwargs)
    network.create_model(**kwargs)
    return network

if __name__ == '__main__':
    if not sys.argv[5]:
        raise RuntimeError('Five arguments required: \n'
                           'loss: loss functions to be used \n'
                           'd: margin enforcement hyperparameter \n'
                           'x_var: variance of input Gaussian noise \n'
                           'Mode: \'train\', \'load\', \'load_all\' or \'interact\' \n'
                           'gpu: number of gpu to be used')
    else:
        gpu = int(sys.argv[5])
        cp.cuda.Device(gpu).use()
        loss = sys.argv[1].lower()
        d = cp.asarray(float(sys.argv[2]), dtype=cp.float32)
        x_var = cp.asarray(float(sys.argv[3]), dtype=cp.float32)
        mode = sys.argv[4].lower()
        
    if mode not in {'train', 'load', 'load_all', 'interact'}:
        raise RuntimeError('Mode should be \'train\', \'load\', \'load_all\' or \'interact\'. You entered {mode} \n'
                           'train: carries out training given settings in exp_settings.py file\n'
                           'load: loads and carries out measurements of trained nets with architecture defined in exp_settings.py AND loss and hyperparameters given as arguments\n'
                           'load_all: loads and carries out measurements of all trained nets with architecture defined in exp_settings.py\n')
        
    input_params = {'loss': loss, 'd': d, 'x_var': x_var, 'gpu': gpu}
    
    while True:
        kwargs = settings(input_params)
        if mode == 'train':
            print('### You entered training mode')
            args = experiment_routine(**kwargs)
            sys.exit("Exiting training")

        elif mode in {'load', 'load_all', 'interact'}:
            print('### You entered load mode. In this mode the robustness measurements will be carried for the specified trained NN')
            # Load trained nets from trained nets folder for specified dataset and architecture file so that the
            # robustness and performance metrics can be computed
            aux = 0
            while True:
                aux = input('Are the dataset {} and architecture {} correct? [y] Yes [n] No\n'.format(kwargs['dataset'], kwargs['arch']))
                if aux.lower() == 'n':
                    sys.exit("Exiting loading. Please input the correct dataset and architecture in \n'exp_settings.py\n'")
                elif aux.lower() == 'y':
                    break

            while True:
                _, curr_fold = os.path.split(kwargs['net_save_dir'])
                curr_dir, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
                measures_save_dir = curr_dir + "/measurements/{}/".format(kwargs['dataset']) + "{}/{}/".format(kwargs['arch'],curr_fold)
                print(measures_save_dir)
                if not os.path.isdir(measures_save_dir[0:-1]):
                    os.makedirs(measures_save_dir[0:-1])
                
                curr_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
                filelist = list_nets(kwargs['net_save_dir'], loss = loss, d = d, x_var = x_var, load_mode = mode)
                print(filelist)
                # loops through all files in the .csv file
                for file in filelist:
                    net_path = file
                    print(kwargs['net_save_dir'])
                    print(measures_save_dir)
                    print(file)
                    kwargs_load = {'exp_count': None, 'success': None, 'data_save_dir': None}
                    kwargs = {**kwargs, **kwargs_load}
                    # Builds NNAgent object with architecture settings
                    network = build_net(**kwargs)
                    # Load trained NN into the NNAgent object
                    network.model = unpickle(net_path)
                    # Changes GPU in case the GPU used during training is different from the one used from here
                    change_gpu(network.model, gpu)
                    # Prepares data with same settings as ones used for training
                    network.prepare_data(mode='load', **kwargs)

                    # Data used for measurements are TEST DATA. Here it changes it to Variable object so that
                    # differentiation w.r.t. inputs can be done
                    x_m = Variable(network.te_x)
                    target = Variable(network.te_y)

                    # Parses the trained NN settings type of loss, x_var, d, training epochs and training number from the file name
                    # code.interact(local=locals())
                    loss = re.search('loss_(.*)_ep', net_path).group(1)
                    epochs = re.search('_ep_(.*)_x_var', net_path).group(1)
                    x_var = re.search('x_var_(.*)_d', net_path).group(1)
                    d = re.search('{}_d_(.*)_'.format(x_var), net_path).group(1)
                    training_num = float(re.search('_d_{}_(.*)'.format(d), net_path).group(1))
                    x_var = float(x_var)
                    d = float(d)
                    
                    arch = network.model.arch
                    network.loss = loss
                    network.epochs = epochs
                    network.x_var = x_var
                    network.d = d
                    network.model.intvl_in = network.intvl_in
                    network.model.center_in = network.center_in

                    # Sets the Bjorck orthogonalization settings for each layer
                    for j in range(len(network.model)):
                        if hasattr(network.model[j], 'W'):
                            network.model[j].config['dynamic_iter'] = True
                            network.model[j].dynamic_iter = True
                    
                    network.model.ortho_iter_red()
                    
                    if mode == 'interact':
                        code.interact(banner="Loaded in interactive mode:", local=locals())
                    
                    # Performance and robustness measurements
                    robust_msr = True
                    num_int = False
                    sample_est = False
                    rs_cr = False
                    robustness.measurements(network, x_m, target, measures_save_dir, robustness = robust_msr, num_int = num_int, sample_est = sample_est, rs_cr = rs_cr)

                sys.exit("Finished the data collection of robustness measurements")