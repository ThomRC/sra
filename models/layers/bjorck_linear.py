"""
Code from Cem Anil adapted to be used with Chainer, and added the iter_red function that dynamically changes the Bjorck iterations:
https://github.com/cemanil/LNets/blob/master/lnets/models/layers/dense/bjorck_linear.py

Cem Anil, James Lucas, Roger Grosse. "Sorting Out Lipschitz Function Approximation"
Proceedings of the 36th International Conference on Machine Learning (2019)
"""
import chainer.functions as F
from chainer import utils
from models.layers.denselinear import DenseLinear
from utils.bjorck_ortho import bjorck_orthonormalize, get_safe_bjorck_scaling, min_ortho_iter
from optimizers.updater_adam import Adam

class BjorckLinear(DenseLinear, Adam):
    """
    A linear layer with Bjorck orthogonalization capability.

    This class extends the functionality of a linear layer by incorporating Bjorck orthogonalization during the forward pass, allowing for improved numerical stability and convergence.

    Args:
        x: The input data to the linear layer.
        n_batch_axes: The number of batch axes. Default is 1.
        train: A flag indicating whether to perform orthogonalization for gradient computation or use pre-orthogonalized weights.

    Returns:
        The result of multiplying the orthogonalized weight matrix with the input data.
    """    
    def __init__(self, in_size, config=None, **kwargs):
        DenseLinear.__init__(self, in_size, **kwargs)
        Adam.__init__(self, **kwargs)
        self.config = config
        self.ortho_w = None
        self.iter = self.config['iter']
        self.dynamic_iter = config['dynamic_iter']

    def forward(self, x, n_batch_axes: int = 1, train = False):
        """
        Carries the forward pass of the linear layer subject to Bjorck orthogonalization

        Args:
            x:
            n_batch_axes: The number of batch axes. The default is 1.
            train: if train = True then the orthogonalization is carried from the original matrix so that the gradients can be computed. If train = False then it used the saved orthogonalized weight matrix

        Returns: the multiplication of the orthogonalized matrix and the previous layer activation
        """
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)

        if train:
            self.orthonormalize()        
        return F.linear(x, self.ortho_w, self.b, n_batch_axes=n_batch_axes)

    def orthonormalize(self):
        """
        Calls the Bjorck orthogonalization function
        """
        if self.config['safe_scaling']:
            # Scale the values of the matrix to make sure the singular values are less than or equal to 1
            scaling = get_safe_bjorck_scaling(self.W)
        else:
            scaling = 1.0

        self.ortho_w = bjorck_orthonormalize(self.W / scaling,
                                             beta=self.config['beta'],
                                             iters=self.iter,
                                             order=self.config['order'])
        return

    def iter_red(self):
        """
        Reduces the number of iterations for achieving an approximately orthogonal layer.

        This function adjusts the iteration count based on the minimum number required to achieve an approximately orthogonal weight matrix for the current layer.
        """        
        # Call the function min_ortho_iter for the current layer, which obtains the minimal number of iterations required to have an approximately orthogonal layer
        self.iter = min_ortho_iter(self.W,
                                        beta=self.config['beta'],
                                        iters=self.iter,
                                        order=self.config['order'])
        print(self.iter)