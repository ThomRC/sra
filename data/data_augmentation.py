import numpy as np
import chainer.functions as F

def randomhorizontalflip(img, p=0.5):
    """
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        img (PIL Image or Tensor): Image to be flipped.
        p (float): probability of the image being flipped. Default value is 0.5
        
    Returns:
        PIL Image or Tensor: Randomly flipped image.
    """
    if np.random.rand(1) < p:
        return F.flip(img, axis = -1)
    return img