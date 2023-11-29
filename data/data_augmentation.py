import numpy as np
import cupy as cp
import chainer.functions as F

def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (ndarray Variable): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    n, _, h, w = img.shape
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = cp.random.randint(0, h - th + 1, size=(1,))
    j = cp.random.randint(0, w - tw + 1, size=(1,))
    return i, j, th, tw

def randomcrop(img, size, padding=None, fill=0, padding_mode="constant"):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    """
    Args:
        img (PIL Image or Tensor): Image to be cropped.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if padding is not None:
        img = F.pad(img, ((0,0),(0,0),(padding,padding),(padding,padding)), padding_mode, constant_values = fill)

    i, j, h, w = get_params(img, (size, size))

    return F.get_item(img, (Ellipsis, slice(i, i + h), slice(j, j + w)))

def randomhorizontalflip(img, p=0.5):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    """
    Args:
        img (PIL Image or Tensor): Image to be flipped.

    Returns:
        PIL Image or Tensor: Randomly flipped image.
    """
    if np.random.rand(1) < p:
        return F.flip(img, axis = -1)
    return img