
"""
Code from Qiyang Li just adapted to be used with Chainer:
https://github.com/ColinQiyangLi/LConvNet/blob/master/lconvnet/layers/utils.py

Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger Grosse, Jörn-Henrik Jacobsen. "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks"
33rd Conference on Neural Information Processing Systems (NeurIPS 2019)
"""
import chainer.functions as F
from utils import convolution_2d

def cyclic_pad_2d(x, pads, even_h=False, even_w=False):
    """
    Implemenation of cyclic padding for 2-D image input
    """
    pad_change_h = -1 if even_h else 0
    pad_change_w = -1 if even_w else 0
    pad_h, pad_w = pads
    if pad_h != 0:
        v_pad = F.concat((x[..., :, -pad_h:, :], x,
                           x[..., :, :pad_h+pad_change_h, :]), axis=-2)
    elif pad_change_h != 0:
        v_pad = F.concat((x, x[..., :, :pad_change_h, :]), axis=-2)
    else:
        v_pad = x
    if pad_w != 0:
        h_pad = F.concat((v_pad[..., :, :, -pad_w:],
                           v_pad, v_pad[..., :, :, :pad_w+pad_change_w]), axis=-1)
    elif pad_change_w != 0:
        h_pad = F.concat((v_pad, v_pad[..., :, :, :+pad_change_w]), axis=-1)
    else:
        h_pad = v_pad
    return h_pad

def conv2d_cyclic_pad(x, weight, bias=None):
    """
    Implemenation of cyclic padding followed by a normal convolution
    """
    kh, kw = weight.shape[-2], weight.shape[-1]
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if len(x.shape) == 3:
        x = F.expand_dims(x, 0)
    # return F.convolution_2d(x, weight, bias)
    return convolution_2d.convolution_2d(x, weight, bias)