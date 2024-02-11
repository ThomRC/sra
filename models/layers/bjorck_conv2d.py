# import chainer
from chainer import as_variable
from chainer.functions.connection import convolution_2d
from chainer import memory_layouts

from models.layers.base_conv import BaseConv2D
from utils.bjorck_ortho import bjorck_orthonormalize, get_safe_bjorck_scaling, min_ortho_iter
from optimizers.updater_adam import Adam

# Implementation from Anil et al. (2019)
# Here the filters are mutually orthonormal, but the convolution layer isn't orthogonal, i.e., it isn't norm preserving

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class BjorckConv2D(BaseConv2D, Adam):
    def __init__(self, in_channels, out_channels, config=None, **kwargs):
        BaseConv2D.__init__(self, in_channels, out_channels, **kwargs)
        Adam.__init__(self, **kwargs)
        self._set_config(config)
        self.ortho_w = None
        self.iter = config['iter']
        self.dynamic_iter = config['dynamic_iter']

        self.original_shape = self.weight.shape

        if self.stride == 1 or self.stride == [1, 1]:
            print("BEWARE: Norm is not being preserved due to stride > 1.  ")

    def forward(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if not hasattr(self, 'iter'):
            self.iter = self.config['iter']

        # x = chainer.as_variable(x)
        x = as_variable(x)
        assert x.layout == self.x_layout
        # self.W can be a Variable instead of Parameter: #8462
        if self.W.raw_array is None:
            _, c, _, _ = memory_layouts.get_semantic_shape(
                x, assumed_layout=self.x_layout)
            self._initialize_params(c)

        if train and self.iter > 0:
            # Applies BO only in training steps, saving the currently orthonormalized weights into self.ortho_w
            self.orthonormalize()
        elif train and self.iter == 0:
            self.ortho_w = self.W
        elif self.dynamic_iter:
            # In the testing step during the training it computes the min_ortho_iter
            # If dynamic_iter = True adapts every 10 epochs the number of iterations layerwise for BO to avoid unnecessary iterations

            if self.last_epoch % 10 == 0 and self.last_epoch > 0:
                self.iter = min_ortho_iter(self.W,
                                           beta=self.config['beta'],
                                           iters=self.iter,
                                           order=self.config['order'])
                print(self.iter)

        return convolution_2d.convolution_2d(
            x, self.ortho_w, self.b, self.stride, self.pad, dilate=self.dilate,
            groups=self.groups, cudnn_fast=self.cudnn_fast)

    def orthonormalize(self):
        """ Calls the Bjorck orthogonalization function """
        
        flattened_weights = self.conv_form_to_matrix_form(self.W, (self.out_channels, -1))

        # Orthonormalize. The scaling makes sure the singular values of the matrix are constrained by 1.
        if self.config['safe_scaling']:
            # Scale the values of the matrix to make sure the singular values are less than or equal to 1
            scaling = get_safe_bjorck_scaling(flattened_weights)
        else:
            scaling = 1.0
        
        ortho_weight_flattened = bjorck_orthonormalize(flattened_weights.T / scaling,
                                                beta=self.config['beta'],
                                                iters=self.iter,
                                                order=self.config['order']).T
        # CHECK IF NEEDS TO BE TRANSPOSED OR NOT FOR ORTHONORMALIZATION

        # Reshape back.
        self.ortho_w = self.matrix_form_to_conv_form(ortho_weight_flattened, self.original_shape)



