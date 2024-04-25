
"""
Implementation of Invertible Downsampling from https://arxiv.org/pdf/1802.07088.pdf

Code from Qiyang Li just adapted to be used with Chainer:
https://github.com/ColinQiyangLi/LConvNet/blob/master/lconvnet/layers/invertible_downsampling.py

Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger Grosse, Jörn-Henrik Jacobsen. "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks"
33rd Conference on Neural Information Processing Systems (NeurIPS 2019)
"""
from chainer import link
from einops import rearrange

class InvertibleDownsampling2d(link.Link):
    """
    Implements 2D invertible downsampling using rearrangement of input data.

    This class performs downsampling on 2D input data by rearranging the data based on the specified kernel size.
    """    
    def __init__(self, kernel_size):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        return rearrange(
            x,
            "b c (w k1) (h k2) -> b (c k1 k2) w h",
            k1=self.kernel_size[0],
            k2=self.kernel_size[1],
        )
