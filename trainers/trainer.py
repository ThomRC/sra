from chainer import cuda, Variable, no_backprop_mode
import cupy as cp
import numpy as np
import six
import csv
import time
import datetime
import pickle
# Import the class to build the model and train the net
from data_processing import mnist_preprocessing, cifar10_preprocessing, cifar100_preprocessing

class NNAgent(object):
    def __init__(self, test_data = 10000, **kwargs):
        super().__init__()
        # Experiment and training settings
        self.data_dir = kwargs['data_save_dir']
        self.net_dir = kwargs['net_save_dir']
        self.nhl = len(kwargs['hl_units'])
        self.exp_num = kwargs['exp_count']

        # Dataset parameters and setup
        self.dataset = kwargs['dataset']
        self.train_data = kwargs['tr_size']
        self.test_data = test_data
        self.intvl_in = kwargs['in_interval']
        self.center_in = kwargs['in_center']
        self.batch_size = kwargs['batch_size']
        self.n_epoch = kwargs['n_epoch']
        self.prepare_data(**kwargs)
        kwargs['n_batch'] = self.M

        # Loss parameters and setup
        self.loss = kwargs['loss']
        self.d = kwargs['d']
        self.x_var = kwargs['x_var']
        self.loss_func, self.smooth_loss, self.margin_loss = self.import_loss()

        # NN creation
        self.init = kwargs['init']
        agent.create_model(**kwargs)

        # Lists that will contain collected training data
        self.acc_train = []
        self.acc_test = []
        self.acc_test_noise = []
        self.loss_value = []

    def import_loss(self):
        """

        Returns:

        """
        from sra-paper.models.utils.selections import select_loss_function
        return select_loss_function(self.loss)

    def prepare_data(self,mode = 'train', **kwargs):
        """ Method to prepare the training and test data

        Args:
            mode: train samples randomly the data for training and test, load uses the indices saved to have the same
            data from training for training and test data
        """
        print("#### Preparing dataset".format(kwargs['dataset']))
        print('Dataset: {} with {} training samples'.format(self.dataset, self.train_data))

        if mode == 'train':
            if self.dataset == "MNIST":
                self.n_out = 10
                self.linf_att_max = 0.35
                self.l2_att_max = 4.6
                self.acc_th = 0.7
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = mnist_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'])
            elif self.dataset == "cifar10":
                self.n_out = 10
                self.linf_att_max = 0.0366
                self.l2_att_max = 0.46
                self.acc_th = 0.3
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar10_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'])
            elif self.dataset == "cifar100":
                self.n_out = 100
                self.linf_att_max = 0.0366
                self.l2_att_max = 0.46
                self.acc_th = 0.1
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar100_preprocessing(self.train_data, self.test_data, self.intvl_in, self.center_in, norm = kwargs['normalization'])
            else:
                print("Asked dataset ", self.dataset, " is not recognized.")
                return
        elif mode == 'load':
            if self.dataset == "MNIST":
                self.n_out = 10
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = mnist_preprocessing(60000, 10000, self.intvl_in, self.center_in, self.model.tr_idx, self.model.te_idx, norm = kwargs['normalization'])
            elif self.dataset == "cifar10":
                self.n_out = 10
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar10_preprocessing(50000, 10000, 4., 0., self.model.tr_idx, self.model.te_idx, norm = kwargs['normalization'])
            elif self.dataset == "cifar10":
                self.n_out = 100
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar100_preprocessing(50000, 10000, 4., 0., self.model.tr_idx, self.model.te_idx, norm = kwargs['normalization'])
            else:
                print("Asked dataset ", self.dataset, " is not recognized.")
                return
        print('Data prepared on {} mode'.format(mode))
        print("#### Finished data preparation")
        self.n_train = self.tr_x.shape[0]
        self.M = self.n_train / float(self.batch_size)
        print("############# Number of epochs:", self.n_epoch)

    def create_model(self, **kwargs):
        """ Method responsible to create the net and set the parameters
        """
        # Network layer sizes
        args = kwargs.pop('hl_units')
        self.model = py38_MLP.FeedForwardNN(self.n_out, *args, **kwargs).to_gpu()
        self.model.n_in = self.tr_x.shape[1]
        # Add zero-padding to tr and te inputs in order to match the HL1 dimension
        if kwargs['in_padding'] and self.tr_x.shape[1] < args[0]:
            print("With dimension matching zero-padding of input")
            self.model.zero_cat_in = True
            delta_dim = args[0] - self.tr_x.shape[1]
            self.tr_x = self.model.xp.hstack((self.tr_x, self.model.xp.zeros((self.tr_x.shape[0],delta_dim))))
            self.te_x = self.model.xp.hstack((self.te_x, self.model.xp.zeros((self.te_x.shape[0],delta_dim))))

    # Method to run the training routine
    def training(self):
        t1 = time.perf_counter()

        # Allocates as chainer variables the validation and training data
        self.valid_in = Variable(self.te_x)
        self.valid_out = Variable(self.te_y)
        self.train_in = Variable(self.tr_x)
        self.train_out = Variable(self.tr_y)

        idx = cp.arange(10000)
        steps = 1

        ##### Epoch loop
        with no_backprop_mode():
            # first call of model without generatic computational graph to initialize values
            self.model(self.train_in)
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

                in_data = self.train_in[data_indices]
                t = self.train_out[data_indices]
                ##### Gradient update averaging loop
                ##### calculate the grads
                self.model.cleargrads()
                ##### Forward the input data from the minibatch in the network

                if self.smooth_loss:
                    ### regular smooth loss
                    _, mean_s, var_s, mean_h, var_h = self.model.moment_propagation(len(self.model), in_data,self.x_var,in_data.array, w_grad=True)
                else:
                    mean_s = self.model(in_data)
                    var_s = None
                loss_out = loss(mean_s, var_s, t, self.d, None, None, x_var = self.x_var)
                ##### Get the gradient of the loss function with respect to each parameter in the network
                loss_out.backward()
                loss_avg += loss_out.array
                update_result = self.model.update_net(epoch)
                steps += 1

            ##### Train accuracy of the updated network after 1 epoch
            accuracy = self.model.validation(self.train_in, self.train_out, train=False)
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

            if (epoch + 1) % 20 == 0:
                accuracy = self.model.validation(self.valid_in, self.valid_out)
                acc_noise = self.model.validation(self.valid_in, self.valid_out, noise_in=True, sd = 1., train=False)

                if (epoch + 1) % 100 == 0 and accuracy < self.acc_th:
                    # In case of reaching epoch 100 and the test accuracy is too low it stops the training to try again
                    return epoch

                if (epoch + 1) % 60 == 0 and (epoch + 1) > 0:
                    print('Learning rate is {}'.format(self.model[0].lr))
                    scale = 1.0
                    cw_cut=0
                    layer=2
                    loss_func='CW' ## 'crossentropy', 'CW', 'SCW', 'sm1' 'snmse', 'smape'
                    num_steps=200
                    # Every 60 epochs obtain the l_inf PGD robustness
                    step_norm='sign' ## 'sign', 'inf', '2'
                    eps_norm='inf' ## 'inf', '2'
                    adv_acc_linf = []
                    samples = 5
                    adv_acc_linf.append(cp.ones(samples) * accuracy)
                    radii_arr = np.linspace(0,self.linf_att_max,15)
                    for j in range(14):
                        radius = radii_arr[j+1]
                        x_var = ((radius/4)**2).astype('float32')
                        t1 = time.perf_counter()
                        aux = cp.zeros(samples)
                        for i in range(samples):
                            step_size = 0.08 * radius
                            t1_1 = time.perf_counter()
                            x_adv = self.model.projected_gradient_descent(self.valid_in, self.valid_out,num_steps=num_steps, step_size=step_size,eps=radius, clamp = (self.center_in - self.intvl_in/2,self.center_in + self.intvl_in/2), eps_norm=eps_norm,step_norm=step_norm, loss_func=loss_func, pert_scale=scale, cw_cut = cw_cut, x_var = x_var)
                            aux[i] = self.model.validation(Variable(x_adv), self.valid_out, noise_in=False, train=False)
                            print("Accuracy for Linf radius = {} is {} #{}".format(radius, aux[i],i))
                            t2_1 = time.perf_counter()
                            print("Exe. time = {}".format(t2_1 - t1_1))
                        adv_acc_linf.append(aux)

                    # Every 60 epochs obtain the l_2 PGD robustness
                    step_norm='2' ## 'sign', 'inf', '2'
                    eps_norm='2' ## 'inf', '2'
                    adv_acc_l2 = []
                    samples = 5
                    adv_acc_l2.append(cp.ones(samples) * accuracy)
                    radii_arr = np.linspace(0,self.l2_att_max,15)
                    for j in range(14):
                        radius = radii_arr[j+1]
                        x_var = ((radius/4)**2).astype('float32')
                        t1 = time.perf_counter()
                        aux = cp.zeros(samples)
                        for i in range(samples):
                            step_size = 0.08 * radius
                            t1_1 = time.perf_counter()
                            x_adv = self.model.projected_gradient_descent(self.valid_in, self.valid_out,num_steps=num_steps, step_size=step_size,eps=radius, clamp = (self.center_in - self.intvl_in/2,self.center_in + self.intvl_in/2), eps_norm=eps_norm,step_norm=step_norm, loss_func=loss_func, pert_scale=scale, cw_cut = cw_cut, x_var = x_var)
                            aux[i] = self.model.validation(Variable(x_adv), self.valid_out, noise_in=False, train=False)
                            print("Accuracy for L2 radius = {} is {} #{}".format(radius, aux[i],i))
                            t2_1 = time.perf_counter()
                            print("Exe. time = {}".format(t2_1 - t1_1))
                        adv_acc_l2.append(aux)

                    np.save(self.dir + '/{}_loss_linf_adv_acc_x_var_{}_d_{}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    np.save(self.dir + '/{}_loss_l2_adv_acc_x_var_{}_d_{}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_l2)))

            print("Epoch: {} Epoch time: {}".format((epoch  + 1), time.time() - now))

        return epoch

    # Method to save the graphs and data into specified folder
    def save_data(self, save = True):
        """

        Args:
            save:

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

    def save_net(self, save = True):
        """ Saves the NNAgent object ''network'' from the current training to the ''save_dir'' directory under the
        ''net_name'' name. If ''save_data'' is 1 then the data collected during the training is also saved CHECK LATER

        Args:
            save:

        """
        if save:
            print("#### Saving trained net in the specified directory")
            self.model.tr_idx = self.tr_idx
            self.model.te_idx = self.te_idx
            self.model.epoch = epoch_aux + 1
            self.model.training_num = 1
            trained_net_name = "/trained_loss_{}_ep_{}_x_var_{}_d_{}_{}".format(self.loss, self.n_epoch, self.x_var,
                                                                                   self.d, self.model.training_num)
            while os.path.isfile(dest_dir + trained_net_name):
                self.model.training_num += 1
                trained_net_name = "/trained_loss_{}_ep_{}_x_var_{}_d_{}_{}".format(self.loss, self.n_epoch, self.x_var,
                    self.d, self.model.training_num)

            with (open(self.net_dir + trained_net_name, "wb") as fdata):
                pickle.dump(self.model, fdata)
        else:
            print("#### Trained net not saved")

def run_training(**kwargs):
    cp.cuda.Device(kwargs['gpu']).use()
    cuda.set_max_workspace_size(33554432)

    start = time.perf_counter()
    agent = NNAgent(test_data = cp.maximum(10000, 60000 - kwargs['tr_size']), **kwargs)#.to_gpu()
    epoch_aux = agent.training()
    agent.save_data(kwargs['save_data'])

    if epoch_aux + 1 == agent.n_epoch:
        print("#### Finished training {}! Training duration was: {}".format(kwargs['exp_count'],
                                                                            time.perf_counter() - start))
        agent.save_net(agent.model.training_num, kwargs['save_net'])
        return 1 # training succeeded
    elif epoch_aux is None:
        return 0
    else:
        print("#### Training not succeeded. Trying again with same settings.")
        return 0 # training not succeeded