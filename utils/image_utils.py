def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Horizontally flipped image.
    """
    """
    def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-1)
    """
    return F_t.hflip(img)

def get_dimensions(img: Tensor) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image dimensions.
    """

    return F_pil.get_dimensions(img)

def pad(img: Tensor, padding: List[int], fill: Union[int, float] = 0, padding_mode: str = "constant") -> Tensor:
    """Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
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

    Returns:
        PIL Image or Tensor: Padded image.
    """
    
    return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    return F_t.crop(img, top, left, height, width)


