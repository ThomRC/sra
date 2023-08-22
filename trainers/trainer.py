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

# To plot
import seaborn as sns
import matplotlib.pyplot as plt

class NNAgent(object):
    def __init__(self, test_data = 10000, **kwargs):
        super().__init__()
        self.dataset = kwargs['dataset']
        self.train_data = kwargs['tr_size']
        self.test_data = test_data
        self.intvl_in = kwargs['in_interval']
        self.center_in = kwargs['in_center']
        self.batch_size = kwargs['batch_size']
        self.n_epoch = kwargs['n_epoch']
        self.dir = kwargs['save_dir']
        self.nhl = len(kwargs['hl_units'])
        self.exp_num = kwargs['exp_count']
        self.smooth_loss = False
        self.margin_loss = False
        self.acc_test = []
        self.acc_test_noise = []
        self.acc_train = []
        self.likelihood = []
        self.totloss = []
        self.grad_norm = []
        self.S_cov_ev = []
        self.smooth_loss = False
        self.margin_loss = False
        self.loss_func = None

    def import_loss(self):
        from sra-paper.models.utils.selections import select_loss_function

         self.loss_func, self.smooth_loss, self.margin_loss = select_loss_function(self.loss)

    # Method to prepare the training and test data
    def prepare_data(self,mode = 'train', **kwargs):
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
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar10_preprocessing(50000, 10000, 4., 0., self.model.tr_idx, norm = kwargs['normalization'])
            elif self.dataset == "cifar10":
                self.n_out = 100
                self.tr_x, self.te_x, self.tr_y, self.te_y, self.tr_idx, self.te_idx = cifar100_preprocessing(50000, 10000, 4., 0., self.model.tr_idx, norm = kwargs['normalization'])
            else:
                print("Asked dataset ", self.dataset, " is not recognized.")
                return
        print('Data prepared on {} mode'.format(mode))
        self.n_train = self.tr_x.shape[0]
        self.M = self.n_train / float(self.batch_size)
        print("############# Number of epochs:", self.n_epoch)

    # Method responsible to create the net and set the parameters
    def set_model_parameter(self, **kwargs):
        # Network layer sizes
        args = kwargs.pop('hl_units')
        kwargs['n_batch'] = self.M
        self.model = py38_MLP.FeedForwardNN(self.n_out, *args, **kwargs).to_gpu()
        self.init = kwargs['init']
        self.model.n_in = self.tr_x.shape[1]
        self.loss = kwargs['loss']
        self.import_loss()
        self.d = kwargs['d']
        self.x_var = kwargs['x_var']
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

        mean_s = None
        var_s = None
        mean_s2 = None
        var_s2 = None

        ##### Epoch loop
        with no_backprop_mode():
            # first call of model without generatic computational graph to initialize values
            self.model(self.train_in)
        print("Total epochs for training = ", self.n_epoch)
        for epoch in range(self.n_epoch):
            dest = '/dock/thomas/vol/trained_nets_MNIST/'
            now = time.time()
            perm_tr = np.random.permutation(self.n_train) # creates a random permutation of n_train integers, where n_train is the number of training examples
            accuracy = 0.
            lkhd_loss_avg = cp.asarray(0.)
            count = 0
            avg_loss_grad_sum = 0
            avg_lik_grad_sum = 0
            avg_lik_grad_ratio_sum = 0

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
                    ### annealed smooth loss (using the same cos-annealing from the adam update)
                else:
                    mean_s = self.model(in_data)
                likelihood = loss(mean_s, var_s, t, self.d, mean_s2, var_s2, x_var = self.x_var)

                lkhd_loss_avg += likelihood.array
                ##### Get the gradient of the loss function with respect to each parameter in the network
                likelihood.backward()

                update_result = self.model.update_net(epoch)

                ### ANNEALING TRAINING

                steps += 1

            if self.test_data != 10000:
                idx = cp.random.choice(self.test_data, 10000, replace=False)

            ##### Train accuracy of the updated network after 1 epoch
            accuracy = self.model.validation(self.train_in, self.train_out, train=False)
            self.acc_train.append(accuracy)
            print("Accuracy train data: {}".format(accuracy))
            ##### Test accuracy of the updated network after 1 epoch
            accuracy = self.model.validation(self.valid_in[idx], self.valid_out[idx], train=False)
            self.acc_test.append(accuracy)
            print("Accuracy test data: {}".format(accuracy))
            ##### Test accuracy of the updated network after 1 epoch with noisy input
            accuracy = self.model.validation(self.valid_in[idx], self.valid_out[idx], noise_in=True, sd = self.intvl_in, train=False)
            self.acc_test_noise.append(accuracy)
            print("Accuracy test data (noisy): {}".format(accuracy))

            self.likelihood.append(lkhd_loss_avg/self.M)
            print("Avg. likelihood loss: {}".format(lkhd_loss_avg / self.M))

            if (epoch + 1) % 20 == 0:
                accuracy = self.model.validation(self.valid_in[idx], self.valid_out[idx])
                acc_noise = self.model.validation(self.valid_in[idx], self.valid_out[idx], noise_in=True, sd = 1., train=False)

                if (epoch + 1) % 100 == 0 and accuracy < self.acc_th:
                    return epoch

                if (epoch + 1) % 60 == 0 and (epoch + 1) > 0:
                    print('Learning rate is {}'.format(self.model[0].lr))
                    scale = 1.0
                    cw_cut=0
                    layer=2
                    loss_func='CW' ## 'crossentropy', 'CW', 'SCW', 'sm1' 'snmse', 'smape'
                    num_steps=200
                    step_norm='sign' ## 'sign', 'inf', '2'
                    eps_norm='inf' ## 'inf', '2'
                    adv_acc_linf = []
                    noise_adv_acc_linf = []
                    samples = 5
                    adv_acc_linf.append(cp.ones(samples) * accuracy)
                    noise_adv_acc_linf.append(cp.ones(samples) * acc_noise)
                    radii_arr = np.linspace(0,self.linf_att_max,15)
                    for j in range(14):
                        radius = radii_arr[j+1]
                        x_var = ((radius/4)**2).astype('float32')
                        t1 = time.perf_counter()
                        aux = cp.zeros(samples)
                        aux2 = cp.zeros(samples)
                        for i in range(samples):
                            step_size = 0.08 * radius
                            t1_1 = time.perf_counter()
                            x_adv = self.model.projected_gradient_descent(self.valid_in[idx], self.valid_out[idx],num_steps=num_steps, step_size=step_size,eps=radius, clamp = (self.center_in - self.intvl_in/2,self.center_in + self.intvl_in/2), eps_norm=eps_norm,step_norm=step_norm, loss_func=loss_func, pert_scale=scale, cw_cut = cw_cut, x_var = x_var)
                            aux[i] = self.model.validation(Variable(x_adv), self.valid_out[idx], noise_in=False, train=False)
                            print("Accuracy for Linf radius = {} is {} #{}".format(radius, aux[i],i))
                            aux2[i] = self.model.validation(Variable(x_adv), self.valid_out[idx], noise_in=True, sd = 1., train=False)
                            print("Noise accuracy for Linf radius = {} is {} #{}".format(radius, aux2[i],i))
                            t2_1 = time.perf_counter()
                            print("Exe. time = {}".format(t2_1 - t1_1))

                        adv_acc_linf.append(aux)
                        noise_adv_acc_linf.append(aux2)

                    step_norm='2' ## 'sign', 'inf', '2'
                    eps_norm='2' ## 'inf', '2'
                    adv_acc_l2 = []
                    noise_adv_acc_l2 = []
                    samples = 5
                    adv_acc_l2.append(cp.ones(samples) * accuracy)
                    noise_adv_acc_l2.append(cp.ones(samples) * acc_noise)
                    radii_arr = np.linspace(0,self.l2_att_max,15)
                    for j in range(14):
                        radius = radii_arr[j+1]
                        x_var = ((radius/4)**2).astype('float32')
                        t1 = time.perf_counter()
                        aux = cp.zeros(samples)
                        aux2 = cp.zeros(samples)
                        for i in range(samples):
                            step_size = 0.08 * radius
                            t1_1 = time.perf_counter()
                            x_adv = self.model.projected_gradient_descent(self.valid_in[idx], self.valid_out[idx],num_steps=num_steps, step_size=step_size,eps=radius, clamp = (self.center_in - self.intvl_in/2,self.center_in + self.intvl_in/2), eps_norm=eps_norm,step_norm=step_norm, loss_func=loss_func, pert_scale=scale, cw_cut = cw_cut, x_var = x_var)
                            aux[i] = self.model.validation(Variable(x_adv), self.valid_out[idx], noise_in=False, train=False)
                            print("Accuracy for L2 radius = {} is {} #{}".format(radius, aux[i],i))
                            aux2[i] = self.model.validation(Variable(x_adv), self.valid_out[idx], noise_in=True, sd = 1., train=False)
                            print("Noise accuracy for L2 radius = {} is {} #{}".format(radius, aux2[i],i))
                            t2_1 = time.perf_counter()
                            print("Exe. time = {}".format(t2_1 - t1_1))

                        adv_acc_l2.append(aux)
                        noise_adv_acc_l2.append(aux2)

                    # print('notclip_SM2_loss_alpha=1_lr={}_{}_epoch{}'.format(self.model[0].lr,self.exp_num, epoch  + 1))
                    # np.save(self.dir + '/notclip_SM2_loss_alpha=1_lr={}_{}_epoch{}'.format(self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # print('SM2_loss_zero_limit_alpha=1_d={}_lr={}_{}_epoch{}'.format(self.d, self.model[0].lr,self.exp_num, epoch  + 1))
                    # np.save(self.dir + '/SM2_loss_zero_limit_alpha=1_d={}_lr={}_{}_epoch{}'.format(self.d, self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    print('{}_loss_x_var={}_d={}_lr={}_{}_epoch{}'.format(self.loss,self.x_var,self.d, self.model[0].lr,self.exp_num, epoch  + 1))
                    # np.save(self.dir + '/{}_loss_alpha=1_d={}_lr={}_{}_epoch{}'.format(self.loss,self.d, self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    np.save(self.dir + '/{}_loss_linf_adv_acc_x_var_{}_d={}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    np.save(self.dir + '/{}_loss_linf_noise_adv_acc_x_var_{}_d={}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_linf)))
                    np.save(self.dir + '/{}_loss_l2_adv_acc_x_var_{}_d={}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_l2)))
                    np.save(self.dir + '/{}_loss_l2_noise_adv_acc_x_var_{}_d={}_{}_epoch{}'.format(self.loss,self.x_var, self.d, self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_l2)))
                    # print('MC_hinge_loss_lr={}_{}_epoch{}'.format(self.model[0].lr,self.exp_num, epoch  + 1))
                    # np.save(self.dir + '/MC_hinge_loss_BO_1_d={}_linf_adv_acc_CW_lr={}_{}_epoch{}'.format(self.d, self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # np.save(self.dir + '/MC_hinge_loss_linf_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # np.save(self.dir + '/MC_hinge_loss_linf_noise_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_linf)))
                    # np.save(self.dir + '/MC_hinge_loss_l2_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_l2)))
                    # np.save(self.dir + '/MC_hinge_loss_l2_noise_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_l2)))
                    # np.save(self.dir + '/SCE_loss_BO_1_iter_linf_adv_acc_CW_lr={}_{}_epoch{}'.format(self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # print('SCE_loss_lr={}_{}_epoch{}'.format(self.model[0].lr,self.exp_num, epoch  + 1))
                    # np.save(self.dir + '/SCE_loss_alpha=1_d={}_lr={}_{}_epoch{}'.format(self.d, self.model[0].lr,self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # np.save(self.dir + '/SCE_loss_linf_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_linf)))
                    # np.save(self.dir + '/SCE_loss_linf_noise_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_linf)))
                    # np.save(self.dir + '/SCE_loss_l2_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(adv_acc_l2)))
                    # np.save(self.dir + '/SCE_loss_l2_noise_adv_acc_BO_30_{}_epoch{}'.format(self.exp_num, epoch  + 1), cuda.to_cpu(cp.asarray(noise_adv_acc_l2)))

            if (epoch + 1) in {20,60,140}:
                print("Reducing x var:",self.x_var)
                print("Reducing d:",self.d)

            print("Epoch: {} Epoch time: {}".format((epoch  + 1), time.time() - now))

        return epoch

    # Method to save the graphs and data into specified folder
    def save_data(self):
        trn_error = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_train))
        tst_error = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_test))
        tst_error_noise = 100 - 100 * cuda.to_cpu(cp.asarray(self.acc_test_noise))

        lklhd = cuda.to_cpu(cp.asarray(self.likelihood))
        tot_loss = lklhd

        file_names = ['trn_error', 'tst_error', 'tst_error_noise', 'likelihood']
        arrays = [trn_error, tst_error, tst_error_noise, lklhd]
        np.set_printoptions(threshold=np.inf)
        for name, data in zip(file_names, arrays):
            csvfile = self.dir + "/{}.csv".format(name)
            with open(csvfile, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in data:
                    writer.writerow([val])

        # Save images of the plotted graphs
        arrays = [trn_error, tst_error, tst_error_noise, lklhd, tot_loss]
        y_axis = ['Error (%)', 'Error (%)', 'Error (%)', 'Error (%)', 'Error (%)', 'Likelihood loss', 'Total loss']
        title = ['Error of train data', 'Error of test data',
                 'Error of test data with noise', 'Evolution of likelihood loss', 'Evolution of total loss']
        file_names = ['trn_error', 'tst_error', 'tst_error_noise', 'likelihood',  'tot_loss']

        for y_name, title_name, data, name in zip(y_axis, title, arrays, file_names):
            x = np.arange(len(data)) + 1
            sns.set_style("darkgrid")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('{}'.format(title_name), fontsize=15)
            if y_name in {'Norm', 'Total loss'}:
                ax.set_xlabel('# Minibatch', fontsize=12)
            else:
                ax.set_xlabel('# Epoch', fontsize=12)
            ax.set_ylabel('{}'.format(y_name), fontsize=12)
            plt.plot(x, data)
            plt.savefig(self.dir+'/{}.png'.format(name), bbox_inches='tight')
            plt.close()

def run_sim(**kwargs):
    save_data = kwargs['save_data']
    save_net = save_data
    cp.cuda.Device(kwargs['gpu']).use()
    cuda.set_max_workspace_size(33554432)

    agent = NNAgent(test_data = cp.maximum(10000, 60000 - kwargs['tr_size']), **kwargs)#.to_gpu()

    print("#### Preparing {} dataset".format(kwargs['dataset']))
    agent.prepare_data(**kwargs)
    print("#### Finished data preparation!")

    agent.set_model_parameter(**kwargs)
    start = time.perf_counter()
    epoch_aux = agent.training()

    if save_data == 1:
        print("####Saving data in the specified directory!")
        agent.save_data()
    else:
        print("####Not saving any data")

    if epoch_aux + 1 == agent.n_epoch:
        print("####Finished training {}! Total running time is: {}".format(
            kwargs['exp_count'],
            time.perf_counter() - start))

        if save_net == 1:
            print("####Saving trained net in the specified directory!")
            agent.model.tr_idx = agent.tr_idx
            agent.model.te_idx = agent.te_idx
            agent.model.epoch = epoch_aux + 1
            agent.model.n_sim = kwargs['success'] + 1

            with open(
                    kwargs['save_dir'] + "/trained_{}_x_var_{}_d_{}_{}".format(datetime.datetime.today().strftime('%Y-%m-%d'),
                                                                     kwargs['x_var'],kwargs['d'],kwargs['success']),
                    "wb") as fdata:
                pickle.dump(agent.model, fdata)
        else:
            print("####Trained net not saved")
        return 1 # succeeded training
    # elif epoch_aux + 1 >= 240:
    #     print("####Finished training {}, but because the training failed. \nTotal running time is: {}".format(
    #         kwargs['exp_count'],
    #         time.perf_counter() - start))
    #     return 1 # succeeded training
    elif epoch_aux is None:
        return 0
    else:
        print("####Training not succeeded. Trying again with same settings.")
        return 0 # not succeeded training