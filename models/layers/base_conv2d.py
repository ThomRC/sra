import chainer
from chainer import initializers
from chainer import link
from chainer import memory_layouts
from chainer.utils import argument
from chainer import variable

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class BaseConv2D(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 bias=False, initialW=None, initial_bias=None, **kwargs):
        super(BaseConv2D, self).__init__()

        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1),
            deterministic='deterministic argument is not supported anymore. '
            'Use chainer.using_config(\'cudnn_deterministic\', value) '
            'context where value is either `True` or `False`.')

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.cudnn_fast = chainer.get_compute_mode() == 'cudnn_fast'
        if self.cudnn_fast:
            x_layout = memory_layouts.CUDNN_CHANNEL_LAST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_LAST_W
        else:
            x_layout = memory_layouts.CUDNN_CHANNEL_FIRST_X
            w_layout = memory_layouts.CUDNN_CHANNEL_FIRST_W

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = int(groups)
        self.x_layout = x_layout

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer, layout=w_layout)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if bias:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)
            else:
                self.b = None

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        if self.out_channels % self.groups != 0:
            raise ValueError('the number of output channels must be'
                             ' divisible by the number of groups')
        if in_channels % self.groups != 0:
            raise ValueError('the number of input channels must be'
                             ' divisible by the number of groups')
        W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
        self.W.initialize(W_shape)

    def _set_config(self, config):
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def conv_form_to_matrix_form(W, matrix_form_shape):
        return W.view(matrix_form_shape)

    @staticmethod
    def matrix_form_to_conv_form(W, conv_form_shape):
        return W.view(conv_form_shape)
