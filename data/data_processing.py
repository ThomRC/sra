import os
import numpy as np
import cupy as cp
from chainer import cuda, datasets
from sklearn.datasets import fetch_openml

def normalize(x, axis):
    """ Function to normalize the image dataset. Each pixel is subtracted by the same pixel mean and divided by the standard deviation"""
    x_m = cp.mean(x, axis = axis)
    x_std = cp.std(x, axis = axis)
    if axis == 0:
        norm_x = cp.nan_to_num((x - x_m.reshape((1, x.shape[1])))/x_std.reshape((1, x.shape[1])))
    elif axis == 1:
        norm_x = cp.nan_to_num((x - x_m.reshape((x.shape[0], 1)))/x_std.reshape((x.shape[0], 1)))
    else:
        print("Not calculated normalized vector. Incorrect axis.")
    return norm_x

def mnist_preprocessing(train_data = 60000, test_data = 10000, intvl_data = 2., in_center = 1., tr_idx = None, te_idx = None, norm = False, arch = 'mlp'):
    """ Iniotializes the MNIST dataset

    Args:
        train_data: size of training data to be used (<= 60000)
        test_data: size of test data to be used ( <= 10000 + 60000 - train_data)
        intvl_data: range of each pixel (e.g., = 2 each pixel can be in the ranges [-1,1] or [0,2])
        in_center: center of the range of each pixel (e.g., if intvl_data = 2 and in_center = 0 range is [-1,1], if intvl_data = 1 and in_center = 1 range is [0.5,1.5],
        tr_idx: when loading a NN this argument is used to use exactly the same training data
        te_idx: when loading a NN this argument is used to use exactly the same test data
        norm:
        arch: 'mlp' or 'cnn'        

    Returns:
        training input
        training output
        test input
        test output
        indices of training data
        indices of test data

    """
    # Goes to sra folder
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra":
        dataset_dir, curr_fold = os.path.split(dataset_dir)

    # Checks if the dataset subfolders exist. If not it creates it
    in_data_dir = dataset_dir + "/datasets/dataset_MNIST/mnist_preprocessed_data_{}_{}.npy".format(intvl_data,in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_MNIST/mnist_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if not os.path.exists(os.path.dirname(in_data_dir)):
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir))
        finally:
            os.umask(original_umask)

    # Checks if the dataset files exist. If not it creates it downloads it using fetch_openml
    if not os.path.exists(in_data_dir):
        mnist = fetch_openml('mnist_784',as_frame=False)
        x = cuda.to_gpu(mnist['data'][:]).astype('float32') / (255. / intvl_data) - (intvl_data / 2) + in_center
        print(in_data_dir)
        cp.save(in_data_dir, x)
        y = cuda.to_gpu(mnist['target'].astype('int32'))
        cp.save(out_data_dir, y)

    x = cp.load(in_data_dir)
    y = cp.load(out_data_dir)

    if norm:
        # Normalizes the input data
        x = normalize(x, axis = 0)
        print("Input successfully normalized")

    if tr_idx is None or te_idx is None:
        # When training a dataset the indices of training and test data doesn't matter, so it creates the indices here
        print("Dataset file to be loaded:")
        print(in_data_dir)
        tr_idx = np.arange(train_data).astype('int32') #int16 = 65536 integers
        te_idx = np.arange(70000 - test_data, 70000).astype('int32')

    # Loads training data and test data from the indices
    tr_x, te_x = x[tr_idx], x[te_idx]
    tr_y, te_y = y[tr_idx], y[te_idx]

    if arch == 'cnn':
        tr_x = tr_x.reshape(tr_x.shape[0], 1, 28, 28)
        te_x = te_x.reshape(te_x.shape[0], 1, 28, 28)

    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx

def cifar10_preprocessing(train_data = 50000, test_data = 10000, intvl_data = 4., in_center = 0., tr_idx = None, norm = False, arch = 'mlp'):
    """ Initializes the Cifar 10 dataset

    Args:
        train_data: size of training data to be used (<= 60000)
        test_data: size of test data to be used ( <= 10000 + 60000 - train_data)
        intvl_data: range of each pixel (e.g., = 2 each pixel can be in the ranges [-1,1] or [0,2])
        in_center: center of the range of each pixel (e.g., if intvl_data = 2 and in_center = 0 range is [-1,1], if intvl_data = 1 and in_center = 1 range is [0.5,1.5],
        tr_idx: when loading a NN this argument is used to use exactly the same training data
        te_idx: when loading a NN this argument is used to use exactly the same test data
        norm:
        arch: 'mlp' or 'cnn'        

    Returns:
        training input
        training output
        test input
        test output
        indices of training data
        indices of test data

    """
    train_data = np.minimum(50000, train_data)
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra":
        dataset_dir, curr_fold = os.path.split(dataset_dir)

    in_data_dir = dataset_dir + "/datasets/dataset_cifar10/cifar10_preprocessed_data_{}_{}.npy".format(intvl_data, in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_cifar10/cifar10_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if not os.path.exists(os.path.dirname(in_data_dir)):
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir))
        finally:
            os.umask(original_umask)

    if not os.path.exists(in_data_dir):
        in_data = np.zeros((60000,3072))
        out_data = np.zeros((60000))
        a = datasets.get_cifar10(ndim = 1)
        for i,j in enumerate(a[0]):
            in_data[i,:] = j[0]
            out_data[i] = j[1]

        for i,j in enumerate(a[1]):
            in_data[i + 50000,:] = j[0]
            out_data[i + 50000] = j[1]
        inputs = cuda.to_gpu(in_data)#.astype(dtype = np.float32))
        labels = cuda.to_gpu(out_data.astype(dtype = np.int32))
        x = inputs * intvl_data - (intvl_data / 2) + in_center
        print(in_data_dir)
        cp.save(in_data_dir, x)
        y = labels
        cp.save(out_data_dir, y)
    te_idx = cp.asarray(range(50000,60000))
    x = cp.load(in_data_dir)
    y = cp.load(out_data_dir)

    if norm:
        x = normalize(x, axis = 0)
        print("Input successfully normalized")

    if tr_idx is None:
        print("Dataset file to be loaded:")
        print(in_data_dir)
        tr_idx = cp.random.choice(50000, train_data, replace=False)

    tr_x, te_x = x[tr_idx,:], x[50000:60000,:]
    tr_y, te_y = y[tr_idx], y[50000:60000]
    tr_x = tr_x.astype(dtype=cp.float32)
    te_x = te_x.astype(dtype=cp.float32)

    if arch == 'cnn':
        tr_x = tr_x.reshape(tr_x.shape[0], 3, 32, 32)
        te_x = te_x.reshape(te_x.shape[0], 3, 32, 32)

    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx

def cifar100_preprocessing(train_data = 50000, test_data = 10000, intvl_data = 4., in_center = 0., tr_idx = None, norm = False, arch = 'mlp'):
    """ Initializes the Cifar 100 dataset

    Args:
        train_data: size of training data to be used (<= 60000)
        test_data: size of test data to be used ( <= 10000 + 60000 - train_data)
        intvl_data: range of each pixel (e.g., = 2 each pixel can be in the ranges [-1,1] or [0,2])
        in_center: center of the range of each pixel (e.g., if intvl_data = 2 and in_center = 0 range is [-1,1], if intvl_data = 1 and in_center = 1 range is [0.5,1.5],
        tr_idx: when loading a NN this argument is used to use exactly the same training data
        te_idx: when loading a NN this argument is used to use exactly the same test data
        norm:
        arch: 'mlp' or 'cnn'

    Returns:
        training input
        training output
        test input
        test output
        indices of training data
        indices of test data

    """
    train_data = np.minimum(50000, train_data)
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra":
        dataset_dir, curr_fold = os.path.split(dataset_dir)

    in_data_dir = dataset_dir + "/datasets/dataset_cifar100/cifar100_preprocessed_data_{}_{}.npy".format(intvl_data, in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_cifar100/cifar100_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if not os.path.exists(os.path.dirname(in_data_dir)):
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir))
        finally:
            os.umask(original_umask)

    if not os.path.exists(in_data_dir):
        in_data = np.zeros((60000,3072))
        out_data = np.zeros((60000))
        a = datasets.get_cifar100(ndim = 1)
        for i,j in enumerate(a[0]):
            in_data[i,:] = j[0]
            out_data[i] = j[1]

        for i,j in enumerate(a[1]):
            in_data[i + 50000,:] = j[0]
            out_data[i + 50000] = j[1]
        inputs = cuda.to_gpu(in_data)#.astype(dtype = np.float32))
        labels = cuda.to_gpu(out_data.astype(dtype = np.int32))
        x = inputs * intvl_data - (intvl_data / 2) + in_center
        print(in_data_dir)
        cp.save(in_data_dir, x)
        y = labels
        cp.save(out_data_dir, y)
    te_idx = cp.asarray(range(50000,60000))
    x = cp.load(in_data_dir)
    y = cp.load(out_data_dir)

    if norm:
        x = normalize(x, axis = 0)
        print("Input successfully normalized")

    if tr_idx is None:
        print("Dataset file to be loaded:")
        print(in_data_dir)
        tr_idx = cp.random.choice(50000, train_data, replace=False)

    tr_x, te_x = x[tr_idx,:], x[50000:60000,:]
    tr_y, te_y = y[tr_idx], y[50000:60000]
    tr_x = tr_x.astype(dtype=cp.float32)
    te_x = te_x.astype(dtype=cp.float32)

    if arch == 'cnn':
        tr_x = tr_x.reshape(tr_x.shape[0], 3, 32, 32)
        te_x = te_x.reshape(te_x.shape[0], 3, 32, 32)

    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx