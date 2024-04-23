import glob
import numpy as np

def list_nets(net_save_dir, loss = None, d = 0, x_var = 0, load_mode = 'load_all'):
    """ Parses trained net files for robustness measurements

    Args:
        net_save_dir: directory containing the trained nets to be loaded
        loss: loss function of trained nets to be loaded
        d: hyperparameter d of the trained nets to be loaded
        x_var: hyperparameter x_var of the trained nets to be loaded
        load_mode: 'load_all' computes robustness measurements for all nets in the folder, anything else loads trained nets with specified loss, d and x_var

    Returns: list of all file names to be loaded

    """
    print(net_save_dir + "/trained_loss_{}*_x_var_{}_d_{}_*".format(loss, x_var, d))
    if load_mode == 'load_all':
        print('### Measurements will be carried for all trained nets')
        files = glob.glob(net_save_dir + "/trained*")
    else:
        print('### Measurements will be carried for {} loss, d = {} and x_var = {}'.format(loss, d, x_var))
        files = glob.glob(net_save_dir + "/trained_loss_{}*_x_var_{}_d_{}_*".format(loss, x_var, d))
    return files

def save_measurements(name_str, measurements_dict):
    """ Saves the robustness measurements data into specified folder

    Args:
        name_str: string containing the destination folder and the part of the file name common to all files
        measurements_dict: dictionary containing the measurement name used as ending of the file name and the data itself saved as .npy file

    """
    print("### Saving collected data into .npy files ###")
    for measure_name, measure in measurements_dict.items():
        if measure is not None:
            np.save(name_str + measure_name, measure)

def manage_gpu():
    pass
