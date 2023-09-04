import os
import numpy as np
import cupy as cp
from chainer import cuda
from sklearn.datasets import fetch_openml

def normalize(x, axis):
    x_m = cp.mean(x, axis = axis)
    x_std = cp.std(x, axis = axis)
    if axis == 0:
        norm_x = cp.nan_to_num((x - x_m.reshape((1, x.shape[1])))/x_std.reshape((1, x.shape[1])))
    elif axis == 1:
        norm_x = cp.nan_to_num((x - x_m.reshape((x.shape[0], 1)))/x_std.reshape((x.shape[0], 1)))
    else:
        print("Not calculated normalized vector. Incorrect axis.")
    return norm_x

def mnist_preprocessing(train_data = 60000, test_data = 10000, intvl_data = 2., in_center = 1., tr_idx = None, te_idx = None, norm = False):
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra-paper":
        dataset_dir, curr_fold = os.path.split(curr_path)

    in_data_dir = dataset_dir + "/datasets/dataset_MNIST/mnist_preprocessed_data_{}_{}.npy".format(intvl_data,in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_MNIST/mnist_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if os.path.exists(os.path.dirname(in_data_dir)) == False:
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir), mode=0o770)
        finally:
            os.umask(original_umask)
    print(in_data_dir)
    if os.path.exists(in_data_dir) == False:
        mnist = fetch_openml('mnist_784',as_frame=False)
        x = cuda.to_gpu(mnist['data'][:]).astype('float32') / (255. / intvl_data) - (intvl_data / 2) + in_center
        print(in_data_dir)
        cp.save(in_data_dir, x)
        y = cuda.to_gpu(mnist['target'].astype('int32'))
        cp.save(out_data_dir, y)

    x = cp.load(in_data_dir)
    y = cp.load(out_data_dir)

    if norm:
        x = normalize(x, axis = 0)
        print("Input successfully normalized")

    if tr_idx is None or te_idx is None:
        tr_idx = np.arange(train_data).astype('int32') #int16 = 65536 integers
        te_idx = np.arange(70000 - cuda.to_cpu(test_data), 70000).astype('int32')

    tr_x, te_x = x[tr_idx], x[te_idx]
    tr_y, te_y = y[tr_idx], y[te_idx]
    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx

def cifar10_preprocessing(train_data = 50000, test_data = 10000, intvl_data = 4., in_center = 0., tr_idx = None, norm = False):
    train_data = np.minimum(50000, train_data)
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra-paper":
        dataset_dir, curr_fold = os.path.split(curr_path)

    in_data_dir = dataset_dir + "/datasets/dataset_cifar10/cifar10_preprocessed_data_{}_{}.npy".format(intvl_data, in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_cifar10/cifar10_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if os.path.exists(os.path.dirname(in_data_dir)) == False:
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir), mode=0o770)
        finally:
            os.umask(original_umask)

    print(in_data_dir)
    if os.path.exists(in_data_dir) == False:
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
        tr_idx = cp.random.choice(50000, train_data, replace=False)

    tr_x, te_x = x[tr_idx,:], x[50000:60000,:]
    tr_y, te_y = y[tr_idx], y[50000:60000]
    tr_x = tr_x.astype(dtype=cp.float32)
    te_x = te_x.astype(dtype=cp.float32)
    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx

def cifar100_preprocessing(train_data = 50000, test_data = 10000, intvl_data = 4., in_center = 0., tr_idx = None, norm = False):
    train_data = np.minimum(50000, train_data)
    dataset_dir, curr_fold = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    while curr_fold != "sra-paper":
        dataset_dir, curr_fold = os.path.split(curr_path)

    in_data_dir = dataset_dir + "/datasets/dataset_cifar100/cifar100_preprocessed_data_{}_{}.npy".format(intvl_data, in_center)
    out_data_dir = dataset_dir + "/datasets/dataset_cifar100/cifar100_preprocessed_target_{}_{}.npy".format(intvl_data, in_center)
    if os.path.exists(os.path.dirname(in_data_dir)) == False:
        try:
            original_umask = os.umask(0)
            os.makedirs(os.path.dirname(in_data_dir), mode=0o770)
        finally:
            os.umask(original_umask)
    print(in_data_dir)
    if os.path.exists(in_data_dir) == False:
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
        tr_idx = cp.random.choice(50000, train_data, replace=False)

    tr_x, te_x = x[tr_idx,:], x[50000:60000,:]
    tr_y, te_y = y[tr_idx], y[50000:60000]
    tr_x = tr_x.astype(dtype=cp.float32)
    te_x = te_x.astype(dtype=cp.float32)
    return tr_x,te_x,tr_y,te_y, tr_idx, te_idx