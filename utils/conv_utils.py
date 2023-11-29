"""
Utility functions
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

def conv2d_cyclic_pad(
        x, weight, bias=None):
    """
    Implemenation of cyclic padding followed by a normal convolution
    """
    kh, kw = weight.shape[-2], weight.shape[-1]
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if len(x.shape) == 3:
        x = F.expand_dims(x, 0)
    # return F.convolution_2d(x, weight, bias)
    return convolution_2d.convolution_2d(x, weight, bias)

# def power_iteration(A, init_u=None, n_iters=10, return_uv=True):
#     """
#     Power iteration for matrix
#     """
#     shape = list(A.shape)
#     # shape[-2] = shape[-1]
#     shape[-1] = 1
#     shape = tuple(shape)
#     if init_u is None:
#         u = torch.randn(*shape, dtype=A.dtype, device=A.device)
#     else:
#         assert tuple(init_u.shape) == shape, (init_u.shape, shape)
#         u = init_u
#     for _ in range(n_iters):
#         v = A.moveaxis-1, -2 @ u
#         v /= v.norm(dim=-2, keepdim=True)
#         u = A @ v
#         u /= u.norm(dim=-2, keepdim=True)
#     s = (u.moveaxis-1, -2 @ A @ v).squeeze(-1).squeeze(-1)
#     if return_uv:
#         return u, s, v
#     return s

# # The following two functions are directly taken from https://arxiv.org/pdf/1805.10408.pdf
# def conv_singular_values_numpy(kernel, input_shape):
#     """
#     Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
#     In International Conference on Learning Representations, 2019.
#     """
#     kernel = np.moveaxis(kernel, [2, 3, 0, 1])
#     transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
#     return np.linalg.svd(transforms, compute_uv=False)

# def conv_clip_2_norm_numpy(
#         kernel, input_shape, clip_to, force_same=False, complex_conv=False,
#         returns_full_conv=False):
#     """
#     Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
#     In International Conference on Learning Representations, 2019.
#     """
#     kernel = np.moveaxis(kernel, [2, 3, 0, 1])
#     transform_coefficients = np.fft.fft2(kernel, input_shape, axes=[0, 1])
#     U, D, V = np.linalg.svd(transform_coefficients,
#                             compute_uv=True, full_matrices=False)
#     if force_same:
#         D_clipped = np.ones_like(D) * clip_to
#     else:
#         D_clipped = np.minimum(D, clip_to)
#     if kernel.shape[2] > kernel.shape[3]:
#         clipped_transform_coefficients = np.matmul(U, D_clipped[..., None] * V)
#     else:
#         clipped_transform_coefficients = np.matmul(U * D_clipped[..., None, :], V)
#     clipped_kernel = np.fft.ifft2(clipped_transform_coefficients, axes=[0, 1])
#     if not complex_conv:
#         clipped_kernel = clipped_kernel.real
#     if not returns_full_conv:
#         clipped_kernel = clipped_kernel[np.ix_(*[range(d) for d in kernel.shape])]
#     clipped_kernel = np.moveaxis(clipped_kernel, [2, 3, 0, 1])
#     return clipped_kernel