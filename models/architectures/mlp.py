import typing as tp  # NOQA

import chainer
from chainer import no_backprop_mode, cuda
import chainer.functions as F
from chainer.functions import relu as activation
import chainer.links as L
from chainer import Variable
# dummy
import losses as losses
import cupy as cp
import numpy as np

class FeedForwardNN(chainer.ChainList):

    """Linear layer with random Normal weights (a.k.a.\\  stochastic Normal fully-connected layer).
    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a parameter 1 ``W`` matrix and optionally a parameter 2 ``P2`` matrix and
    a bias vector ``b`` as parameters.
    ``W`` and ``P2`` hold the parameters describing the Normal distribution from which
    the weight matrix ``W``, used for the forward pass, is sampled from.
    If ``initial_w`` is left to the default value of ``None``, the weight matrix
    ``W`` is initialized with i.i.d. Gaussian samples, each of which has zero
    mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The bias vector ``b``
    is of size ``out_size``. If the ``initial_bias`` is to left the default
    value of ``None``, each element is initialized as zero.  If the ``nobias``
    argument is set to ``True``, then this link does not hold a bias vector.
    Args:
        n_in (int or None): Dimension of input vectors. If unspecified or
            ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        n_out (int): Dimension of output vectors. If only one value is
            passed for ``in_size`` and ``out_size``, that value will be used
            for the ``out_size`` dimension.
        parametrization (str): Name of the parametrization used to sample the
            W matrix.
            ``usual``, ``natural``, ``det``
        nobias (bool): If ``True``, then this function does not use the bias.
        initial_w (:ref:`initializer <initializer>`): Initializer to initialize
            the weight matrix mean. W and P2 are obtained according to their
            relatioship to the mean. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2. If ``initial_w`` is ``None``, then the
            weights are initialized with i.i.d. Gaussian samples, each of which
            has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
    """
    def __init__(
            self,
            n_in: tp.Optional[int],
            *args, # tuple containing the number of units in each HL
            n_out: tp.Optional[int] = None,
            batchnorm: tp.Optional[bool] = False,
            fc_hl: tp.Optional[int] = 2,
            **kwargs
    ) -> None:
        super(FeedForwardNN, self).__init__()
        if n_out is None:
            n_in, n_out = None, n_in
        self.n_in = n_in
        self.n_out = n_out
        self.units = args
        self.zero_cat_in = False
        self.n_hl = fc_hl
        self.epoch = 0
        self.n_sim = kwargs['n_sim']

        if kwargs['ortho_ws'] is True:
            from bjorck_linear import BjorckLinear as LinearLink
            kwargs['beta'] = 0
            print('Training with *deterministic* and orthogonal weights without rotation')
        else:
            from chainer.links import Linear as LinearLink
            kwargs['beta'] = 0
            print('Training with *deterministic* and orthogonal weights with rotation')

        if not all(isinstance(x, int) for x in args):
            raise RuntimeError('Number of units in a layer must all be integers.')

        if self.n_hl != len(args):
            raise RuntimeError('Number of layers and specified number of units per layer does not match.')

        with self.init_scope():
            self.nobias = kwargs['nobias']
            self.bjorck_config = kwargs['bjorck_config']

            if self.n_hl > 0:
                for i in range(fc_hl):
                    if kwargs['ortho_ws']:
                        self.add_link(LinearLink(args[i], i, config = self.bjorck_config, **kwargs).to_gpu())
                    else:
                        self.add_link(LinearLink(args[i], i, **kwargs).to_gpu())

                    if batchnorm:
                        self.add_link(L.BatchNormalization(size = args[i]).to_gpu())
            self.add_link(LinearLink(n_out,  fc_hl, config = self.bjorck_config, **kwargs).to_gpu())

    def __call__(self, x, y = None, norm_out = False, train = True):
        h = x
        for link in range(len(self) - 1):
            if hasattr(self[link], 'W'):
                h = activation(self[link](h, train = train))
            else:
                h = self[link](h)
        h = self[link + 1](h, train = train)

        return h

    def grad_norm(self):
        lw_grad_norm = self.xp.array([])
        mean_abs_grad = self.xp.array([])
        tot_grad = self.xp.array([])
        count = 0
        for link in self:
            if hasattr(link, 'W'):
                mean_abs_grad = self.xp.hstack((mean_abs_grad,link.mean_abs_grad()))
                tot_grad = self.xp.hstack((tot_grad,link.W.grad.flatten()))
                lw_grad_norm = self.xp.hstack((lw_grad_norm, self.xp.linalg.norm(link.W.grad.flatten())))
                count += 1
        tot_grad_norm = self.xp.linalg.norm(tot_grad)
        return lw_grad_norm, mean_abs_grad, tot_grad_norm

    def update_net(self, epoch):
        lw_grad_norm, mean_abs_grad, tot_grad_norm = self.grad_norm()
        for _, link in enumerate(self):
            if hasattr(link, 'W'):
                update_result = link.update(epoch, tot_grad_norm)
                if update_result == 0:
                    return 0
        return 1

    def noise_inj(self, x, sd, dist = 'Normal', radius = 0.3):
        if dist == 'Normal':
            x_noise = self.xp.random.standard_normal(x.shape).astype(self.xp.float32) * sd
        elif dist == 'Uniform':
            x_noise = (self.xp.random.rand(x.shape).astype(self.xp.float32)*2 - 1) * radius
        return x_noise

    def validation(self, val_in, val_out, noise_in = False, sd = 1., dist = 'Normal', train = True, ssnr = False, ssnr_ratio = None):
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        with no_backprop_mode():
            if noise_in == True and ssnr == False:
                noise = self.noise_inj(val_in, sd, dist)
            elif noise_in == True and ssnr == True:
                if ssnr_ratio is None:
                    raise RuntimeError('Need to specify SSNR ratio parameter')
                else:
                    noise = self.noise_inj(val_in, self.intvl_in * 6/255, 'Normal') + self.center_in

            if noise_in == False and train and ssnr == False:
                if len(val_out) > 10000:
                    tr_spls = 10000
                else:
                    tr_spls = len(val_out)
                idx = self.xp.random.choice(len(val_out), tr_spls, replace=False)
                with no_backprop_mode():
                    val_out = val_out[idx]
                    h = val_in[idx]
            elif noise_in == False and not train and ssnr == False:
                h = val_in
            elif noise_in == True and ssnr == False:
                h = val_in + noise
            elif noise_in == True and ssnr == True:
                h = ssnr_ratio * val_in + (1 - ssnr_ratio) * noise

            for link in range(len(self) - 1):
                if hasattr(self[link], 'W'):
                    # h = F.relu(self[link](h))
                    h = activation(self[link](h, train = train))
                else:
                    h = self[link](h)
            h = self[link + 1](h, train = train)

            return F.accuracy(h, val_out).array

    def emp_prob(self, x, target, sd, samples = 100, dist = 'Normal', radius = 0.3):
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        with no_backprop_mode():
            count_vec = self.xp.zeros(len(target))
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
        # Using no_backprop_mode method since it is just required to forward the input through the network, and not to build a computational graph to apply the backprop
        idx_aux = self.xp.arange(len(target))
        not_c_mat = self.xp.zeros((len(target),10), dtype='bool')
        not_c_mat[idx_aux, target.array] = True
        not_c_mat = self.xp.logical_not(not_c_mat)
        with no_backprop_mode():
            mean_c = self.xp.zeros(len(target))
            mean_sq_c = self.xp.zeros(len(target))
            mean_max_notc = self.xp.zeros(len(target))
            mean_sq_max_notc = self.xp.zeros(len(target))
            for _ in range(samples):
                noise = self.noise_inj(x, sd, dist, radius)
                h = x + noise

                for link in range(len(self) - 1):
                    if hasattr(self[link], 'W'):
                        h = activation(self[link](h))
                    else:
                        h = self[link](h)

                h = self[link + 1](h)

                not_tgt_out = h.array[not_c_mat].reshape((len(target), -1))
                mean_c += h.array[idx_aux, target.array]
                mean_sq_c += h.array[idx_aux, target.array]**2
                mean_max_notc += not_tgt_out.max(axis=1)
                mean_sq_max_notc += not_tgt_out.max(axis=1)**2

            return (mean_c - mean_max_notc) / samples, ((mean_sq_c / samples - (mean_c / samples) ** 2) + (mean_sq_max_notc / samples - (mean_sq_max_notc / samples) ** 2)) * samples / (samples - 1)

    def out_sampling(self, x, sd, samples = 1000, dist = 'Normal'):
        with no_backprop_mode():
            h_samples = []

            for _ in range(samples):
                h = x + self.noise_inj(x, sd, dist)

                for link in range(len(self) - 1):
                    if hasattr(self[link], 'W'):
                        h = activation(self[link](h))
                    else:
                        h = self[link](h)

                h = self[link + 1](h)
                h_samples.append(h.array.T)
            return h_samples

    def projected_gradient_descent(self, x, y, num_steps, step_size, step_norm, eps, eps_norm,
                                   clamp=(0,1), y_target=None, loss_func = 'crossentropy', layer = None, x_var = 0, pert_scale = 1, cw_cut = 0):
        """Performs the projected gradient descent attack on a batch of images."""
        x_adv = Variable(self.xp.copy(x.array))
        targeted = y_target is not None
        num_channels = x.shape[1]
        delta_x = self.xp.random.uniform(-eps*pert_scale, eps*pert_scale, x_adv.array.shape).astype('float32')
        x_adv = Variable(self.xp.clip(x_adv.array + delta_x, a_max = clamp[1], a_min = clamp[0]))
        for i in range(num_steps):
            self.cleargrads()
            _x_adv = Variable(x_adv.array)
            if loss_func == 'crossentropy':
                loss = losses.sce(self(_x_adv, train = False), y_target if targeted else y)
            elif loss_func == 'CW':
                loss = losses.cw_loss(self(_x_adv, train = False), y, cut = cw_cut)
            elif loss_func == 'snmse':
                if layer is None or layer == 0 or layer > self.n_hl + 1:
                    raise RuntimeError('Please choose the layer to calculate snmse from 1 to L+1, where L is the # of hidden layers')
                h_clean, mean_s, var_s, mean_h, var_h = self.moment_propagation(layer, _x_adv, x_var, x.array, clean_actv = True)
                loss = F.sum(losses.snmse(mean_s, var_s, h_clean.array))
            elif loss_func == 'smape':
                if layer is None or layer == 0 or layer > self.n_hl + 1:
                    raise RuntimeError('Please choose the layer to calculate smape from 1 to L+1, where L is the # of hidden layers')
                h_clean, mean_s, var_s, mean_h, var_h = self.moment_propagation(layer, _x_adv, x_var, x.array, clean_actv = True)
                loss = F.sum(losses.smape(mean_s, var_s, h_clean.array))
            elif loss_func == "SCW":
                _, mean_s, var_s, mean_h, var_h = self.moment_propagation(len(self), _x_adv, x_var, x.array)
                loss = losses.scw1_loss(mean_s, var_s, y)
            elif loss_func == "sm1":
                _, mean_s, var_s, mean_h, var_h = self.moment_propagation(len(self), _x_adv, x_var, x.array)
                loss = losses.sm1_loss(mean_s, var_s, y)
            elif loss_func == "sm2":
                _, mean_s, var_s, mean_h, var_h = self.moment_propagation(len(self), _x_adv, x_var, x.array)
                loss = losses.sm2_loss(mean_s, var_s, y)

            loss.backward()

            with no_backprop_mode():
                # Force the gradient step to be a fixed size in a certain norm
                if step_norm == 'sign':
                    ### This calculates the gradient of FGSM
                    gradients = self.xp.sign(_x_adv.grad/len(y)) * step_size
                elif step_norm == 'inf':
                    ### The projection l-inf normalization is
                    linf_norm_pos = np.linalg.norm(cuda.to_cpu(_x_adv.grad[:, 0:self.n_in]/len(y)), ord = np.inf, axis = 1) > 0
                    max_idx = _x_adv.grad[:, 0:self.n_in].argmax(axis=1)/len(y)
                    max_mat = cuda.to_cpu(self.xp.zeros_like(_x_adv.grad[:, 0:self.n_in]/len(y), dtype = 'bool'))
                    max_mat[linf_norm_pos,cuda.to_cpu(max_idx[linf_norm_pos]).tolist()] = True
                    gradients = self.xp.clip(_x_adv.grad[:, 0:self.n_in]/len(y), a_min = -1, a_max = +1) ## projection to 1-radius l-inf ball
                    gradients[max_mat] = self.xp.sign(gradients[max_mat])  ## if inside the l-inf ball, projection onto the 1-radius l-inf sphere
                    gradients *=  step_size ## scale according to step size
                elif step_norm == '2':
                    grad_norm = self.xp.linalg.norm(_x_adv.grad[:, 0:self.n_in]/len(y), ord = 2, axis = 1).reshape((-1,1))
                    non_zero_idx = grad_norm != 0
                    indices = cp.arange(len(y)).reshape((-1,1))
                    gradients = (_x_adv.grad/len(y)) * step_size
                    gradients[indices[non_zero_idx],:] = gradients[indices[non_zero_idx],:]/grad_norm[non_zero_idx].reshape((-1,1))

                if targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label w.r.t. the image data
                    x_adv.array[:, 0:self.n_in] -= gradients[:, 0:self.n_in]
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t. the model parameters
                    x_adv.array[:, 0:self.n_in] += gradients[:, 0:self.n_in]

            # Project back into l_norm ball and correct range
            if eps_norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                x_adv.array[:, 0:self.n_in] = self.xp.clip(x_adv.array[:, 0:self.n_in], a_max = x.array[:, 0:self.n_in] + eps, a_min = x.array[:, 0:self.n_in] - eps)
            elif eps_norm == '2':
                delta = x_adv.array - x.array
                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                scaling_factor = self.xp.clip(self.xp.linalg.norm(delta.reshape((delta.shape[0], -1)), ord = 2, axis=1), a_min = eps)
                delta *= eps / scaling_factor.reshape((-1, 1))
                x_adv.array = x.array + delta
            x_adv.array = self.xp.clip(x_adv.array, a_min = clamp[0], a_max = clamp[1])

        return x_adv.array

    def moment_propagation(self, layer, x_m, x_var, x_clean, w_grad = False, ortho_ws = True, clean_actv = False):
        h_clean = x_clean
        h_v = self.xp.ones_like(x_m.array)*x_var

        # for link in range(len(self) - 1):
        for link in range(layer):
            if hasattr(self[link], 'W'):
                if self[link].layer_num == 0:
                    mean_s, var_s, h_m, h_v = self[link].relu_moment_propagation(x_m, h_v, ortho_ws = ortho_ws, w_grad = w_grad)
                else:
                    mean_s, var_s, h_m, h_v = self[link].relu_moment_propagation(h_m, h_v, ortho_ws = ortho_ws, w_grad = w_grad)

                if clean_actv:
                    if link+1 != len(self):
                        h_clean = F.relu(self[link].forward(h_clean, train = w_grad))
                    else:
                        h_clean = self[link].forward(h_clean, train = w_grad)

        return h_clean, mean_s, var_s, h_m, h_v