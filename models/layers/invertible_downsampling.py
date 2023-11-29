"""
A wrapper for LipschitzConv2d. It uses invertible downsampling to mimic striding.

Invertible Downsampling is implemented as described in https://arxiv.org/pdf/1802.07088.pdf.
"""
from chainer import link
from einops import rearrange
from utils.conv_utils import cyclic_pad_2d

# Implementation from Li et al. (2019)
# Here the convolution layer is orthogonal, i.e., it is norm preserving

class InvertibleDownsampling2d(link.Link):
    def __init__(self, kernel_size):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        # kh, kw = self.kernel_size[0]//2, self.kernel_size[1]//2
        # if x.shape[2]%self.kernel_size[0] != 0 or x.shape[3]%self.kernel_size[1] != 0:
        #     x = cyclic_pad_2d(x, (kh - x.shape[2]%kh, kw - x.shape[3]%kw), ((kh+1) % 2 == 0), ((kw+1) % 2 == 0))

        return rearrange(
            x,
            "b c (w k1) (h k2) -> b (c k1 k2) w h",
            k1=self.kernel_size[0],
            k2=self.kernel_size[1],
        )



    # def extra_repr(self):
    #     return 'kernel_size={kernel_size}'.format(**self.__dict__)


    # rearrange(x,"b c (w k1) (h k2) -> b (c k1 k2) w h",k1=self.kernel_size[0],k2=self.kernel_size[1],)