import six
import csv
import time
import datetime
import pickle
import os

from chainer import cuda, Variable, no_backprop_mode
import cupy as cp
import numpy as np

from data.data_processing import mnist_preprocessing, cifar10_preprocessing, cifar100_preprocessing
from data.data_augmentation import randomcrop, randomhorizontalflip
import models.architectures.mlp as mlp

class NNAgent(object):
    def __init__(self, test_data = 10000, **kwargs):
        super().__init__()
        # Experiment and training settings
        self.data_dir = kwargs['data_save_dir']
        self.net_dir = kwargs['net_save_dir']
        self.exp_num = kwargs['exp_count']

        # Dataset parameters and setup
        self.dataset = kwargs['dataset']
        self.train_data = kwargs['tr_size']
        self.test_data = test_data
        self.intvl_in = kwargs['in_interval']
        self.center_in = kwargs['in_center']
        self.batch_size = kwargs['batch_size']
        self.n_epoch = kwargs['n_epoch']

        # Loss parameters and setup
        self.loss = kwargs['loss']
        self.d = kwargs['d']
        self.x_var = kwargs['x_var']
        self.loss_func, self.smooth_loss, self.margin_loss = self.import_loss()

        # NN creation setting
        self.init = kwargs['init']

        # Lists that will contain collected training data
        self.acc_train = []
        self.acc_test = []
        self.acc_test_noise = []
        self.loss_value = []

    def import_loss(self):
        """ Import the loss function

        Returns: loss function to be used

        """
        from models.utils.selections import select_loss_function
        return select_loss_function(self.loss)

    def prepare_data(self, mode = 'train', **kwargs):
        """ Method to prepare the training and test data

        Args:
            mode: train samples randomly the data for training and test, load uses the indices saved to have the same
            data from training for training and test data
        """

        if mode == 'train':
            print("#### Preparing dataset".format(kwargs['dataset']))
            print('Dataset: {} with {} training samples'.format(self.dataset, self.train_data))
            if self.dataset == "MNIST":
                self.n_out = 10
                self.linf_att_max = 0.35
                self.l2_att_max = 4.6
                self.acc_th = 0.7
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = mnist_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'], arch = kwargs['arch'])
            elif self.dataset == "cifar10":
                self.n_out = 10
                self.linf_att_max = 0.0366
                self.l2_att_max = 0.46
                self.acc_th = 0.3
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar10_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'], arch = kwargs['arch'])
            elif self.dataset == "cifar100":
                self.n_out = 100
                self.linf_att_max = 0.0366
                self.l2_att_max = 0.46
                self.acc_th = 0.1
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar100_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'], arch = kwargs['arch'])
            else:
                print("Asked dataset ", self.dataset, " is not recognized.")
                return
            print("#### Finished data preparation")
            self.n_train = self.tr_x.shape[0]
            self.M = self.n_train / float(self.batch_size)
            print("############# Number of epochs:", self.n_epoch)
        elif mode == 'load':
            if self.dataset == "MNIST":
                self.n_out = 10
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = mnist_preprocessing(60000, 10000, self.intvl_in, self.center_in, self.model.tr_idx, self.model.te_idx, norm = kwargs['normalization'], arch = kwargs['arch'])
            elif self.dataset == "cifar10":
                self.n_out = 10
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar10_preprocessing(50000, 10000,self.intvl_in, self.center_in, self.model.tr_idx, norm = kwargs['normalization'], arch = kwargs['arch'])
            elif self.dataset == "cifar10":
                self.n_out = 100
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar100_preprocessing(50000, 10000, self.intvl_in, self.center_in, self.model.tr_idx, norm = kwargs['normalization'], arch = kwargs['arch'])
            else:
                print("Asked dataset ", self.dataset, " is not recognized.")
                return

    def create_model(self, **kwargs):
        """ Method responsible to create the net and set the parameters """
        # Network layer sizes
        args = kwargs.pop('fc_hl_units')
        self.model = mlp.FeedForwardNN(self.n_out, *args, **kwargs).to_gpu()
        self.model.n_in = self.tr_x.shape[1]


    def training(self):
        """ Function to run the training part of the training routine

        Returns: the epoch that training stopped to then check if training was successful

        """
        print('### Starting training')
        # Allocates as chainer variables the validation and training data
        self.valid_in = Variable(self.te_x)
        self.valid_out = Variable(self.te_y)
        self.train_in = Variable(self.tr_x)
        self.train_out = Variable(self.tr_y)

        steps = 1

        ##### Epoch loop
        with no_backprop_mode():
        #     # first call of model to initialize values without generating computational graph
        #     self.model(self.train_in, train = False)
            self.model(self.train_in[0:2])
        self.model.ortho_iter_red()
        
        print("Total epochs for training = ", self.n_epoch)
        for epoch in range(self.n_epoch):
            now = time.time()
            perm_tr = np.random.permutation(self.n_train) # creates a random permutation of n_train integers, where n_train is the number of training examples
            loss_avg = cp.asarray(0.)
            count = 0

            # Minibatch loop
            for batch_idx in six.moves.range(0,self.n_train,self.batch_size):
                count += 1
                data_indices = perm_tr[batch_idx:batch_idx + self.batch_size]
                in_data = randomhorizontalflip(self.train_in[data_indices])
                # in_data = randomcrop(self.train_in[data_indices], 32, 4)
                # in_data = randomcrop(randomhorizontalflip(self.train_in[data_indices]), 32, 4)
                # in_data = self.train_in[data_indices]
                
                t = self.train_out[data_indices]
                ##### Gradient update averaging loop
                ##### calculate the grads
                self.model.cleargrads()
                ##### Forward the input data from the minibatch in the network

                if self.smooth_loss:
                    ### regular smooth loss
                    _, mean_s, var_s, mean_h, var_h = self.model.moment_propagation(len(self.model), in_data, self.x_var, w_grad=True)
                else:
                    mean_s = self.model(in_data)
                    var_s = None
                loss_out = self.loss_func(mean_s, var_s, t, self.d, None, None, x_var = self.x_var)
                ##### Get the gradient of the loss function with respect to each parameter in the network
                loss_out.backward()
                loss_avg += loss_out.array
                self.model.update_net(epoch)
                steps += 1

            ##### Train accuracy of the updated network after 1 epoch
            # with no_backprop_mode():
            tr_sample_idx = np.random.choice(50000, 5000, replace=False) # cp -> np
            accuracy = self.model.validation(self.train_in[tr_sample_idx], self.train_out[tr_sample_idx], train=False)
            self.acc_train.append(accuracy)
            print("Accuracy train data: {}".format(accuracy))
            ##### Test accuracy of the updated network after 1 epoch
            accuracy = self.model.validation(self.valid_in, self.valid_out, train=False)
            self.acc_test.append(accuracy)
            print("Accuracy test data: {}".format(accuracy))
            ##### Test accuracy of the updated network after 1 epoch with noisy input
            accuracy = self.model.validation(self.valid_in, self.valid_out, noise_in=True, sd = self.intvl_in, train=False)
            self.acc_test_noise.append(accuracy)
            print("Accuracy test data under noise: {}".format(accuracy))

            self.loss_value.append(loss_avg/self.M)
            print("Avg. loss: {}".format(loss_avg / self.M))

            if (epoch + 1) % 10 == 0:
                self.model.ortho_iter_red()

            if (epoch + 1) % 20 == 0:
                accuracy = self.model.validation(self.valid_in, self.valid_out)

                if (epoch + 1) % 100 == 0 and accuracy < self.acc_th:
                    # In case of reaching epoch 100 and the test accuracy is too low it stops the training to try again
                    return epoch

            print("Epoch: {} Epoch time: {}".format((epoch  + 1), time.time() - now))

        return epoch

    # Method to save the graphs and data into specified folder
    def save_data(self, save = True):
        """ Save the data collected during training for training and test error, test error under noise and loss

        Args:
            save: boolean that informs if data should be saved

        """
        if save:
            print("#### Saving data in the specified directory")
            trn_error = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_train))
            tst_error = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_test))
            tst_error_noise = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_test_noise))
            loss_value = cuda.to_cpu(cp.asarray(self.loss_value))

            file_names = ['trn_error', 'tst_error', 'tst_error_noise', 'loss_value']
            arrays = [trn_error, tst_error, tst_error_noise, loss_value]
            np.set_printoptions(threshold=np.inf)
            for name, data in zip(file_names, arrays):
                csvfile = self.data_dir + "/{}.csv".format(name)
                with open(csvfile, "w") as output:
                    writer = csv.writer(output, lineterminator='\n')
                    for val in data:
                        writer.writerow([val])
        else:
            print("#### Data not saved")

    def save_net(self, epoch, save = True):
        """ Saves the NNAgent object ''network'' from the current training to the ''save_dir'' directory under the
        ''net_name'' name

        Args:
            save: boolean that informs if trained net should be saved

        """
        if save:
            print("#### Saving trained net in the specified directory")
            self.model.tr_idx = self.tr_idx
            self.model.te_idx = self.te_idx
            self.model.epoch = epoch
            self.model.training_num = 1
            trained_net_name = "/trained_loss_{}_ep_{}_x_var_{}_d_{}_{}".format(self.loss, self.n_epoch, self.x_var,
                                                                                   self.d, self.model.training_num)
            while os.path.isfile(self.net_dir + trained_net_name):
                self.model.training_num += 1
                trained_net_name = "/trained_loss_{}_ep_{}_x_var_{}_d_{}_{}".format(self.loss, self.n_epoch, self.x_var,
                    self.d, self.model.training_num)

            with open(self.net_dir + trained_net_name, "wb") as fdata:
                pickle.dump(self.model, fdata)
        else:
            print("#### Trained net not saved")

def run_training(**kwargs):
    """ Runs training routine

    Args:
        **kwargs: dictionary containing all training settings

    Returns: 1 if training is successful, 0 otherwise

    """
    cp.cuda.Device(kwargs['gpu']).use()
    cuda.set_max_workspace_size(33554432)

    start = time.perf_counter()
    agent = NNAgent(test_data = np.maximum(10000, 60000 - kwargs['tr_size']), **kwargs)
    agent.prepare_data(mode = 'train', **kwargs)
    kwargs['n_batch'] = agent.M
    agent.create_model(**kwargs)
    epoch_aux = agent.training()
    agent.save_data(kwargs['save_data'])

    if epoch_aux + 1 == agent.n_epoch:
        print("#### Finished training {}! Training duration was: {}".format(kwargs['exp_count']+1,
                                                                            time.perf_counter() - start))
        agent.save_net(epoch_aux + 1, kwargs['save_net'])
        return 1 # training succeeded
    elif epoch_aux is None:
        return 0
    else:
        print("#### Training not succeeded. Trying again with same settings.")
        return 0 # training not succeeded