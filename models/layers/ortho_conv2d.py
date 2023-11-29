"""
BCOP parameterization with block convolution procedure adapted from the official Tensorflow repo:
https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/init_ops.py#L683
"""
from chainer import link, variable, initializers
import chainer.functions as F
import numpy as np
import cupy as cp
import einops

from optimizers.updater_adam import Adam
from utils.conv_utils import conv2d_cyclic_pad
from utils.bjorck_ortho import bjorck_orthonormalize_bcop, get_safe_bjorck_scaling, min_ortho_iter_bcop

from models.activations.relu_moments import relu_moments

def orthogonal_matrix(n):
    a = cp.random.randn(n, n)
    q, r = cp.linalg.qr(a)
    return q * cp.sign(cp.diag(r))

def symmetric_projection(n, ortho_matrix, mask=None):
    """Compute a n x n symmetric projection matrix.
    Args:
      n: Dimension.
    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = ortho_matrix
    # randomly zeroing out some columns
    if mask is None:
        mask = (cp.random.randn(n) > 0).float()
    c = q * mask
    # return c.mm(c.t())
    return c @ c.T

def block_orth(p1, p2):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    assert p1.shape == p2.shape
    n = p1.shape[0]
    eye = cp.identity(n, dtype=p1.dtype)
    aux = p1 @ p2

    return F.reshape(F.stack([aux, p1 - aux, p2 - aux, eye - p1 - p2 + aux]),(2,2,n,n))

def matrix_conv(m1, m2):
    """Matrix convolution.
    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.
    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = m1[0, 0].shape[0]
    if n != (m2[0, 0]).shape[0]:
        raise ValueError(
            "The entries in matrices m1 and m2 " "must have the same dimensions!"
        )
    
    k = int(np.sqrt(m1.shape[0]*m1.shape[1]))
    l = int(np.sqrt(m2.shape[0]*m2.shape[1]))
    size = k + l - 1 # width and height of final matrix

    # Compute matrix convolution between m1 and m2.
    result = []
    for i in range(size):
        for j in range(size):
            result.append(F.sum(m1[max(0, k - 1 - i):min(k, size - i), max(0, k - 1 - j):min(k, size - j), :, :] @ m2[max(0,l - size + i):min(i + 1,l), max(0,l - size + j):min(j + 1,l), :, :], axis = (0,1)))

    return F.reshape(F.stack(result),(size,size,n,n))

def convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.T
    else:
        ortho = cp.identity(cout)
    if ksize == 1:
        return ortho.reshape(1,1,ortho.shape[0],ortho.shape[1])
    
    p = block_orth(sym_projs[0], sym_projs[1])
    for _ in range(1, ksize - 1):
        p = matrix_conv(p, block_orth(sym_projs[_ * 2], sym_projs[_ * 2 + 1]))
    
    p = ortho.reshape(1,1,ortho.shape[0],ortho.shape[1]) @ p
    
    p = einops.rearrange(p, 'k1 k2 cin cout -> cout cin k2 k1')
    p = p[:, 0:cin, :, :]

    if flipped:
        return F.moveaxis(p, (0,2), (1,3))
    return p

def convolution_orthogonal_generator(ksize, cin, cout, P, Q):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
    ortho = orthogonal_matrix(cout)[0:cin, :] # cin x cout orthogonal matrix, where cin <= cout
    if ksize == 1:
        return ortho.reshape(1,1,ortho.shape[0],ortho.shape[1])

    p = block_orth(symmetric_projection(cout, P[0]), symmetric_projection(cout, Q[0]))
    for _ in range(ksize - 2):
        p = matrix_conv(p, block_orth(symmetric_projection(cout, P[_ + 1]), symmetric_projection(cout, Q[_ + 1])))

    p = ortho.reshape(1,1,ortho.shape[0],ortho.shape[1]) @ p
    p = einops.rearrange(p, '(h k1) (w k2) -> h w k1 k2', k1=ksize, k2=ksize)

    if flipped:
        return p
    return F.moveaxis(p, (0,2), (1,3))

def convolution_orthogonal_initializer(ksize, cin, cout):
    P, Q = [], []
    cmax = max(cin, cout)
    for _ in range(ksize - 1):
        P.append(orthogonal_matrix(cmax))
        Q.append(orthogonal_matrix(cmax))
    P, Q = F.stack(P), F.stack(Q)
    return convolution_orthogonal_generator(ksize, cin, cout, P, Q)

class BCOP(link.Link, Adam):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        bias=False,
        config=None, 
        **kwargs
    ):
        # super().__init__()
        # super(BCOP, self).__init__()
        super().__init__()
        assert stride == 1, "BCOP convolution only supports stride 1."
        assert padding is None or padding == kernel_size // 2, "BCOP convolution only supports k // 2 padding. actual - {}, required - {}".format(padding, kernel_size // 2)

        self.config = config
        self.ortho_w = None
        self.iter = 0
        self.dynamic_iter = config['dynamic_iter']

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_channels = max(self.in_channels, self.out_channels)
        self.num_kernels = 2 * kernel_size - 1
        
        # init_scope is needed for the permanent parameters and variables
        with self.init_scope():
            # Define the unconstrained matrices Ms and Ns for Ps and Qs
            scale = 1.0 / (self.max_channels ** 0.5)
            # initializer = initializers._get_initializer(initializers.Orthogonal(scale=scale))
            ### Note: calling the main parameter tensor as W so that the child class Adam can handle it across different kinds of layers
            # self.W = [variable.Parameter(initializer, (self.max_channels, self.max_channels)) for _ in range(self.num_kernels)]
            self.W = variable.Parameter(cp.asarray([scale * orthogonal_matrix(self.max_channels) for _ in range(self.num_kernels)]).astype(dtype=cp.float32))
            
            # Bias parameters in the convolution
            if bias:
                scale = 1.0 / np.sqrt(self.out_channels)
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initializers.Uniform(scale=scale))
                self.b = variable.Parameter(bias_initializer, self.out_channels)
            else:
                self.b = None  # type: tp.Optional[variable.Variable]

            
            # The mask controls the rank of the symmetric projectors (full half rank).
            self.mask = variable.Variable(cp.concatenate((cp.ones((1, 1, self.max_channels // 2)),
                    cp.zeros((1, 1, self.max_channels - self.max_channels // 2))), axis = -1).astype(dtype=cp.float32)) # CHECK LATER if I need .astype("float32")
    
        Adam.__init__(self, **kwargs)

    def forward(self, x, train = False):
        if train:
            self.orthonormalize()
        elif self.iter == 0:
            self.iter = min_ortho_iter_bcop(self.W,
                                           beta=self.config['beta'],
                                           iters=self.config['iter'],
                                           order=self.config['order'])        
            self.orthonormalize(x)
        elif self.dynamic_iter:
            # If dynamic_iter = True adapts every 10 epochs the number of iterations layerwise for BO to avoid unnecessary iterations
            if self.last_epoch % 10 == 0 and self.last_epoch > 0:
                self.iter = min_ortho_iter_bcop(self.W,
                                           beta=self.config['beta'],
                                           iters=self.iter,
                                           order=self.config['order'])
                print(self.iter)

        # apply cyclic padding to the input and perform a standard convolution
        return conv2d_cyclic_pad(x, self.ortho_w, self.b)

    def orthonormalize(self, x = None):
        """ Calls the orthogonal convolution function """
        if self.config['safe_scaling']:
            # Scale the values of the matrix to make sure the singular values are less than or equal to 1
            scaling = get_safe_bjorck_scaling(self.W)
        else:
            scaling = 1.0

        # orthognoalize all the matrices using Bjorck
        ortho = bjorck_orthonormalize_bcop(self.W / scaling,
                                            beta=self.config['beta'],
                                            iters=self.iter,
                                            order=self.config['order'])

        # compute the symmetric projectors
        H = ortho[0, : self.in_channels, : self.out_channels]
        # H = ortho[0]
        PQ = ortho[1:]
        PQ = PQ * self.mask
        PQ = PQ @ F.moveaxis(PQ, -1, -2)

        # compute the resulting convolution kernel using block convolutions
        self.ortho_w = convolution_orthogonal_generator_projs(self.kernel_size, self.in_channels, self.out_channels, H, PQ)

        return 

    def relu_moment_propagation(self, x_m, x_v, w_grad = False, layer_num = None, x_var = None):
            """ Computes the pre-activation's mean and variance vectors and calls the relu_moments function

            Args:
                x_m: previous layer's mean vector
                x_v: previous layer's variance vector (we assume independent activations, i.e., zero covariances
                w_grad: boolean telling whether the gradients are needed or not

            Returns:

            """
            # if self.W.array is None:
            #     in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            #     self._initialize_params(in_size)

            if w_grad:
                self.orthonormalize() # carries orthonormalization only in the case of needing the W gradients
                W = self.ortho_w
            else:
                W = self.ortho_w.array

            mean_s = conv2d_cyclic_pad(x_m, W, self.b)
            
            if layer_num == 0:
                var_s = variable.Variable(cp.zeros_like(mean_s.array)) + x_var
            else:
                var_s = conv2d_cyclic_pad(x_v, W**2, self.b)

            h_m, h_v = relu_moments(mean_s, var_s)

            return mean_s, var_s, h_m, h_v