import typing as tp  # NOQA
import code
import cupy as cp
import numpy as np
import chainer
from chainer import no_backprop_mode, cuda
import chainer.functions as F
from chainer.functions import relu as activation
import chainer.links as L
from chainer import Variable

import models.losses as losses

class FeedForwardNN(chainer.ChainList):
    """
    Class for feedforward NN that is a list containing in each element the sequence of operations (conv. layer, activation, linear layer, activation,...)
    """
    def __init__(
            self,
            n_in: tp.Optional[int],
            *args, # tuple containing the number of units in each HL
            n_out: tp.Optional[int] = None,
            cv_hl: tp.Optional[int] = 0,
            fc_hl: tp.Optional[int] = 3,
            **kwargs
    ) -> None:
        super(FeedForwardNN, self).__init__()
        if n_out is None:
            n_in, n_out = None, n_in
        self.n_in = n_in
        self.n_out = n_out
        self.units = args
        self.zero_cat_in = False
        self.n_hl = cv_hl + fc_hl
        self.epoch = 0
        self.arch = kwargs['arch']
        self.gpu = kwargs['gpu']
        kwargs['beta'] = 0

        if self.arch == 'cnn':
            if kwargs['dataset'] in {'cifar10','cifar100'}:
                self.in_channels = 3
            else:
                self.in_channels = 1
        else:
            if kwargs['dataset'] in {'cifar10','cifar100'}:
                self.in_channels = 3072
            else:
                self.in_channels = 784

        if self.arch == 'cnn':
            self.channels = kwargs['cv_config']['channel']
            self.ksizes = kwargs['cv_config']['ksize']
            self.strides = kwargs['cv_config']['stride']

        if kwargs['ortho_ws']:
            from models.layers.bjorck_linear import BjorckLinear as LinearLink
            print('NN with orthogonal layers')
            if self.arch == 'cnn':
                from models.layers.ortho_conv2d import BCOP
                from models.layers.invertible_downsampling import InvertibleDownsampling2d
        else:
            from chainer.links import Linear as LinearLink
            print('Training with non-orthogonal weights')

        if not all(isinstance(x, int) for x in args):
            raise RuntimeError('Number of units in a layer must all be integers.')

        print('### Generating NN model')
        print('# If this is the first time running it can take several minutes. Please wait')

        with self.init_scope():
            self.bias = kwargs['bias'] # whether units from each layer have bias or not
            self.bjorck_config = kwargs['bjorck_config']

            if self.n_hl > 0:
                # Creates the object where each element is a linear layer
                for i in range(cv_hl):
                    if kwargs['ortho_ws']:                        
                        if self.strides[i] > 1:
                            assert self.ksizes[i] % self.strides[i] == 0
                            # the invertible downsampling is applied before the convolution
                            self.add_link(InvertibleDownsampling2d(self.strides[i]))
                            # In case stride > 1 need to send the modified parameters to initialize the convolutional layer
                            # compute the kernel size of the actual convolution based on stride                            
                        # compute the in_channels based on stride
                        true_in_channels = self.in_channels * self.strides[i] * self.strides[i]
                        # assert true_in_channels >= self.channels[i] # to guarantee that in case of different channel sizes the norm is preserved in the forward pass
                        true_stride = 1
                        true_kernel_size = self.ksizes[i] // self.strides[i]
                        true_padding = true_kernel_size // 2
                        self.add_link(BCOP(true_in_channels, self.channels[i], true_kernel_size, true_stride, true_padding, config = self.bjorck_config, **kwargs).to_gpu())
                        self.in_channels = self.channels[i]
                    else:
                        self.add_link(Convolutional2d(args[i], **kwargs).to_gpu())

                for i in range(fc_hl):
                    if kwargs['ortho_ws']:
                        self.add_link(LinearLink(args[i], config = self.bjorck_config, **kwargs).to_gpu())
                    else:
                        self.add_link(LinearLink(args[i], **kwargs).to_gpu())
                
            self.add_link(LinearLink(n_out, config = self.bjorck_config, **kwargs).to_gpu())

    def __call__(self, x, y = None, norm_out = False, train = False):
        """
        Carries the forward pass through each element of the NN object under the specified activation function
        """
        h = x
        for link in range(len(self) - 1):
            if hasattr(self[link], 'W'):
                h = self[link](h, train = train)
                h = activation(h)
            else:
                h = self[link](h)
        h = self[link + 1](h, train = train)
        return h

    def ortho_iter_red(self):
        """
        Iterates through the layers of the model to reduce the number of iterations required for Bjorck orthonormalizations.

        This method checks each layer for a weight matrix 'W' and dynamic iteration flag, then calls the 'iter_red' method to reduce iterations if conditions are met.
        """        
        for link in range(len(self)):
            if hasattr(self[link], 'W') and self[link].dynamic_iter:
                self[link].iter_red()

    def grad_norm(self):
        """
        Obtain the l2-norm of the gradient to be used in the update_net function to avoid exploding gradients
        """
        tot_grad = self.xp.array([])
        count = 0
        for link in self:
            if hasattr(link, 'W'):
                tot_grad = self.xp.hstack((tot_grad,link.W.grad.flatten()))
                count += 1
        tot_grad_norm = self.xp.linalg.norm(tot_grad)
        return tot_grad_norm

    def update_net(self, epoch):
        """
        Updates the weights from each layer using the update function from each linear layer object
        """
        tot_grad_norm = self.grad_norm()
        for _, link in enumerate(self):
            if hasattr(link, 'W'):
                link.update(epoch, tot_grad_norm)

    def noise_inj(self, x, sd, dist = 'Normal', radius = 0.3):
        """
        Obtains a noise sample of same dimension as input x to be injected in the input
        """
        if dist == 'Normal':
            x_noise = self.xp.random.standard_normal(x.shape).astype(self.xp.float32) * sd
        elif dist == 'Uniform':
            x_noise = (self.xp.random.rand(x.shape).astype(self.xp.float32)*2 - 1) * radius
        return x_noise

    def validation(self, x, target, noise_in = False, sd = 1., dist = 'Normal', train = True, ssnr = False, ssnr_ratio = None):
        """
        Function that returns the accuracy over the input array 'x' with targets 'target'

        Args:
            x: array of inputs subject to classification
            target: array of target classes
            noise_in: boolean telling whether use noise or not
            sd: standard deviation of the input noise in case noise_in = True
            dist: distribution to be used in case noise_in = True ('Normal' or 'Uniform')
            train: boolean telling whether the input is the training data or not, in case of True and the number of samples > 10000 calculates accuracy for random 10000 samples
            ssnr: boolean telling whether the input will be corrupted by a "gray" image
            ssnr_ratio: proportion of the input that is corrupted by the "gray" image

        Returns: scalar value with the accuracy
        """
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        with no_backprop_mode():
            if noise_in and not ssnr:
                noise = self.noise_inj(x, sd, dist)
            elif noise_in and ssnr:
                if ssnr_ratio is None:
                    raise RuntimeError('Need to specify SSNR ratio parameter')
                else:
                    noise = self.noise_inj(x, self.intvl_in * 6/255, 'Normal') + self.center_in

            if train:
                # Only 10000 samples from training data are used and there's no added noise if train = True
                if len(target) > 10000:
                    tr_spls = 10000
                    idx = np.random.choice(len(target), tr_spls, replace=False)
                else:
                    tr_spls = len(target)
                    idx = np.arange(tr_spls)
                
                with no_backprop_mode():
                    target = target[idx]
                    h = x[idx]
            elif ssnr:
                # If ssnr = True and train = False the input is corrupted by adding a "gray" image
                h = ssnr_ratio * x + (1 - ssnr_ratio) * noise
            elif not noise_in:
                # If train = False and noise_in = False we have the usual test accuracy
                h = x
            elif noise_in:
                # If train = False and noise_in = True we have the test accuracy under noisy input
                h = x + noise


            for link in range(len(self) - 1):
                if hasattr(self[link], 'W'):
                    h = activation(self[link](h, train = train))
                else:
                    h = self[link](h)
            h = self[link + 1](h, train = train) # If train = True it orthogonalizes the weights and we can take the gradients of the weights

            return F.accuracy(h, target).array

    def output_sampling(self, x, sd, samples = 100, dist='Normal', radius=0.3):
        """ 
        Obtains samples of the output layer for a single random input perturbation

        Args:
            x: array of inputs subject to classification
            sd: standard deviation of the isotropic Gaussian noise to be added to the input
            samples: number of samples to be takes for the estimate
            dist: isotropic distribution to be used. Can be either 'Normal' or 'Uniform'
            radius: for the case of 'Uniform', radius is half the interval

        Returns:
            (samples, n_classes) matrix
        """
        
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        out = self.xp.zeros((samples, 10))
        if len(x.shape) == 3:
            x = cp.repeat(cp.expand_dims(x.array,0), samples, 0)
        with no_backprop_mode():
            h = x + self.noise_inj(x, sd, dist, radius)

            for link in range(len(self) - 1):
                if hasattr(self[link], 'W'):
                    h = activation(self[link](h))
                else:
                    h = self[link](h)

            h = self[link + 1](h)
            out = h.array
            return out

    def emp_prob(self, x, target, sd, samples = 100, dist = 'Normal', radius = 0.3):
        """
        Obtains a sample estimate of probability of correct classification

        Args:
            x: array of inputs subject to classification
            target: array of target classes
            sd: standard deviation of the isotropic Gaussian noise to be added to the input
            samples: number of samples to be takes for the estimate
            dist: isotropic distribution to be used. Can be either 'Normal' or 'Uniform'
            radius: for the case of 'Uniform', radius is half the interval

        Returns: the fraction of samples correctly classified
        """
        with no_backprop_mode(): # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
            count_vec = np.zeros(len(target))
            for _ in range(samples):
                noise = self.noise_inj(x, sd, dist, radius)
                h = x + noise

                for link in range(len(self) - 1):
                    if hasattr(self[link], 'W'):
                        h = activation(self[link](h))
                    else:
                        h = self[link](h)

                h = self[link + 1](h)
                corr_out = self.xp.argmax(h.array, axis = 1)
                corr_out_idx = corr_out == target.array
                count_vec[corr_out_idx] += 1

            return count_vec/samples

    def sample_mean_margin(self, x, target, sd, samples = 100, dist='Normal', radius=0.3):
        """
        Obtains a sample estimate of the mean margin, i.e., the smoothed margin

        Args:
            x: array of inputs subject to classification
            target: array of target classes
            sd: standard deviation of the isotropic Gaussian noise to be added to the input
            samples: number of samples to be takes for the estimate
            dist: isotropic distribution to be used. Can be either 'Normal' or 'Uniform'
            radius: for the case of 'Uniform', radius is half the interval

        Returns:
            margin sample mean
            sample margin variance
        """
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        target_cpu = cuda.to_cpu(target.array)
        target_len = len(target_cpu)
        idx_aux = np.arange(target_len)
        not_c_mat = np.zeros((target_len,10), dtype='bool')
        not_c_mat[idx_aux, target_cpu] = True
        not_c_mat = np.logical_not(not_c_mat)
        with no_backprop_mode():
            mean_c = self.xp.zeros(target_len)
            mean_sq_c = self.xp.zeros(target_len)
            mean_max_notc = self.xp.zeros(target_len)
            mean_sq_max_notc = self.xp.zeros(target_len)
            for _ in range(samples):
                noise = self.noise_inj(x, sd, dist, radius)
                h = x + noise

                for link in range(len(self) - 1):
                    if hasattr(self[link], 'W'):
                        h = activation(self[link](h))
                    else:
                        h = self[link](h)

                h = self[link + 1](h)

                not_tgt_out = h.array[not_c_mat].reshape((target_len, -1))
                mean_c += h.array[idx_aux, target_cpu]
                mean_sq_c += h.array[idx_aux, target_cpu]**2
                mean_max_notc += not_tgt_out.max(axis=1)
                mean_sq_max_notc += not_tgt_out.max(axis=1)**2

            return (mean_c - mean_max_notc) / samples, ((mean_sq_c / samples - (mean_c / samples) ** 2) + (mean_sq_max_notc / samples - (mean_sq_max_notc / samples) ** 2)) * samples / (samples - 1)

    def projected_gradient_descent(self, x, target, num_steps, step_size, step_norm, eps, eps_norm,
                                   clamp=(0,1), y_target=None, loss_func = 'crossentropy', x_var = 0, pert_scale = 1, cw_cut = 0):
        """
        Performs the projected gradient descent (PGD) attack on a batch of images
        It carries the PGD attack given a specified number of steps, step norm, attack radius norm, and loss function. Other arguments are described bellow.
        The function accepts three different kinds of losses, but "CW" is the one used for all experiments from the paper
        
        Args:
            x: array of inputs subject to classification
            target: array of target classes
            num_steps: number of steps taken during PGD
            step_size: scaling factor of the adversarial vector to be added to the input
            step_norm: the norm of the PGD attack 'sign', 'inf' and '2'. Note: this is the conjugate norm, i.e., if the attack is in the linf-ball then step norm is 'sign' (1-norm), if the attack is in the l2-ball then '2', if in the l1-ball then 'inf'
            eps: the radius of the lp-ball
            eps_norm: the norm p of the lp-ball
            clamp: the total input space
            y_target: in case of targeted attacks, i.e., instead of increasing the maximal incorrect output it maximizes the target
            loss_func: loss function to be used for the attacks, either 'crossentropy' for SCE, 'CW' for Carlini and Wagner loss, or 'sm' for the smoothed margin
            x_var: variance to be used in case of the 'sm' loss
            pert_scale: the scale of the randomness of the initial 'x_0' around the input. If 0 then all the attacks start exactly at 'x', if 1 then the initial point is drawn uniformly inside of a Uniform cube around the input with same radius as the attack
            cw_cut: the cut to be used for the 'CW' loss

        Returns: a self.xp ndarray with same shape as the input containing the adversarial examples computed with PGD
        """
        import copy
        aux_shape = [x.shape[0]] # holds array that will contain the broadcasted ndarray with first dimension equals to the number of still correctly classified inputs
        for _ in range(len(x.shape)-1): aux_shape.append(1)
        aux_shape2 = copy.copy(aux_shape)
        
        x_adv = Variable(self.xp.copy(x.array))
        targeted = y_target is not None
        delta_x = self.xp.random.uniform(-eps*pert_scale, eps*pert_scale, x_adv.array.shape).astype('float32')
        x_adv = Variable(self.xp.clip(x_adv.array + delta_x, a_max = clamp[1], a_min = clamp[0]))
        for _ in range(num_steps):
            self.cleargrads()
            _x_adv = Variable(x_adv.array)
            if loss_func == 'crossentropy':
                loss = losses.sce(self(_x_adv, train = False), y_target if targeted else target)
            elif loss_func == 'CW':
                loss = losses.cw_loss(self(_x_adv, train = False), target, cut = cw_cut)
            elif loss_func == "sm":
                _, mean_s, var_s, mean_h, var_h = self.moment_propagation(len(self), _x_adv, x_var, train = False)
                loss = losses.sm_loss(mean_s, var_s, target)

            loss.backward()

            with no_backprop_mode():
                # Forces the gradient step to be at most a fixed size in a specified norm step_norm
                if step_norm == 'sign':
                    ### Calculates the gradient for attacks with l_inf ball attacks, i.e., FGSM
                    gradients = self.xp.sign(_x_adv.grad) * step_size
                elif step_norm == 'inf':
                    ### Calculates the gradient for attacks with l_1 ball attacks
                    linf_norm_pos = cp.linalg.norm(_x_adv.grad, ord = cp.inf, axis = 1) > 0
                    max_idx = _x_adv.grad.argmax(axis=1)
                    max_mat = cp.zeros_like(_x_adv.grad, dtype = 'bool')
                    max_mat[linf_norm_pos,max_idx[linf_norm_pos].tolist()] = True
                    gradients = self.xp.clip(_x_adv.grad, a_min = -1, a_max = +1) ## projection to 1-radius l-inf ball
                    gradients[max_mat] = self.xp.sign(gradients[max_mat])  ## if inside the l-inf ball, projection onto the 1-radius l-inf sphere
                    gradients *=  step_size ## scale according to step size
                elif step_norm == '2':
                    ### Calculates the gradient for attacks with l_2 ball attacks                    
                    grad_norm = self.xp.linalg.norm(_x_adv.grad.reshape((_x_adv.shape[0], -1)), ord = 2, axis = 1).reshape((-1, 1))
                    non_zero_idx = grad_norm != 0
                    indices = cp.arange(len(target)).reshape((-1,1))  # cp -> np
                    gradients = (_x_adv.grad) * step_size
                    aux_shape[0] = indices[non_zero_idx].shape[0]
                    gradients[indices[non_zero_idx],:] = gradients[indices[non_zero_idx],:]/grad_norm[non_zero_idx].reshape(aux_shape)

                if targeted:
                    # Targeted attack: Gradient descent with on the loss of the (incorrect) target label w.r.t. the image data
                    x_adv.array -= gradients
                else:
                    # Untargeted attack: Gradient ascent on the loss of the correct label w.r.t. the model parameters
                    x_adv.array += gradients

            # Project back into l_norm ball and correct range
            if eps_norm == 'inf':
                x_adv.array = self.xp.clip(x_adv.array, a_max = x.array + eps, a_min = x.array - eps)
            elif eps_norm == '2':
                delta = x_adv.array - x.array
                # Assume x and x_adv are batched tensors where the first dimension is a batch dimension
                scaling_factor = self.xp.clip(self.xp.linalg.norm(delta.reshape((delta.shape[0], -1)), ord = 2, axis=1), a_min = eps)
                delta *= eps / scaling_factor.reshape(aux_shape2)
                x_adv.array = x.array + delta
            x_adv.array = self.xp.clip(x_adv.array, a_min = clamp[0], a_max = clamp[1])

        return x_adv.array

    def moment_propagation(self, layer, x_m, x_var, w_grad = False, ortho_ws = True, clean_actv = False):
        """
        Carries the moment propagation

        Args:
            layer: the layer until which the moment propagation should be carried (in general the layer should be the output)
            x_m: the mean input, which for the case of isotropic Gaussian input is the 'clean' input
            x_var: the variance of the input Gaussian noise
            w_grad: whether the gradients w.r.t. the weights should be stored. Needed when calling it during training
            ortho_ws: whether Bjorck orthogonalization should be carried
            clean_actv: whether the layer's 'clean' (zero-variance) activation should be stored. For non-linear NNs in general the mean layer activation is different to the 'clean' layer activation. By returning the clean and mean activation you can measure how much each layer deviates under noise

        Returns:
            activation for input with zero-variance ("clean input") of specified hidden layer
            mean of specified layer's pre-activation
            variance of specified layer's pre-activation
            mean of specified hidden layer
            variance of specified hidden layer
        """
        h_clean = x_m.array
        h_v = self.xp.ones_like(x_m.array)*x_var

        for link in range(layer):
            if hasattr(self[link], 'W'):
                if link == 0:
                    mean_s, var_s, h_m, h_v = self[link].relu_moment_propagation(x_m, h_v, w_grad = w_grad, layer_num = 0, x_var = x_var)
                elif link == 1 and not hasattr(self[link - 1], 'W'):
                    mean_s, var_s, h_m, h_v = self[link].relu_moment_propagation(h_m, h_v, w_grad = w_grad, layer_num = 0, x_var = x_var)
                else:
                    mean_s, var_s, h_m, h_v = self[link].relu_moment_propagation(h_m, h_v, w_grad = w_grad)

                if clean_actv:
                    if link+1 != len(self):
                        h_clean = F.relu(self[link](h_clean, train = w_grad))
                    else:
                        h_clean = self[link](h_clean, train = w_grad)
            else:
                if link == 0:
                    h_m = self[link](x_m)
                    h_v = self[link](h_v)
                else:
                    h_m = self[link](h_m)
                    h_v = self[link](h_v)

                if clean_actv:
                    h_clean = self[link](h_clean)

        return h_clean, mean_s, var_s, h_m, h_v