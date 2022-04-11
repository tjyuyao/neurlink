import math
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor
import neurlink.nn as nn
from .nerve import Nerve


class ConvArithmetic:
    """ Convolution arithmetic for dynamic padding derivation.

    This class is an implementation of the following paper and adapted for dilated convolution:
    
        Dumoulin, V. & Visin, F. A guide to convolution arithmetic for deep learning. arXiv:1603.07285 [cs, stat] (2016).
    """

    @staticmethod
    def size(base_size, down_scale):
        return base_size // down_scale + base_size % down_scale

    @staticmethod
    def padding(i, o, k, d, s):
        return (d * k + (o - 1) * s - d - i + 2) // 2

    @staticmethod
    def padding_transposed(it, ot, k, d, s):
        p = ConvArithmetic.padding(ot, it, k, d, s)
        a = ot - ((it - 1) * s - 2 * p + d * (k - 1) + 1)
        return p, a


class ConvNd(Nerve):
    def __init__(
        self,
        spatial_dims,
        kernel_size,
        dilation=1,
        groups=1,
        bias=False,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        padding_mode: str = "zeros",
        transposed: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        in_channels, in_scale = self.input_links[0]
        out_channels, out_scale = self.target_shapes[0]

        if out_scale:
            stride = out_scale // in_scale
            assert stride >= 1
        else:
            stride = 1

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")


        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, kernel_size, kernel_size),
                **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.norm = norm(out_channels)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
        padding = (
            kernel_size * dilation // 2
        )  # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
