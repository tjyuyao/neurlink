import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import neurlink.nn as nn
import neurlink.nn.functional as F
import torch
from torch import Tensor

from .common_types import NeurlinkAssertionError
from .nerve import Nerve, Shape, ShapeSpec
from .utils import expect_int, is_sequence_of, ntuple, reverse_repeat_tuple


class ConvArithmetic:
    """Convolution arithmetic for dynamic padding derivation.

    This class is an implementation of the following paper and adapted for dilated convolution:

        Dumoulin, V. & Visin, F. A guide to convolution arithmetic for deep learning. arXiv:1603.07285 [cs, stat] (2016).

    Also check the documentation of PyTorch nn.Conv2d and nn.Conv2dTransposed.
    """

    @staticmethod
    def size(base_size, down_scale) -> int:
        return base_size // down_scale + base_size % down_scale

    @staticmethod
    def padding(i, o, k, d, s) -> int:
        return (d * k + (o - 1) * s - d - i + 2) // 2

    @staticmethod
    def padding_transposed(it, ot, k, d, s) -> int:
        p = ConvArithmetic.padding(ot, it, k, d, s)
        a = ot - ((it - 1) * s - 2 * p + d * (k - 1) + 1)
        return p, a

    @staticmethod
    def stride(larger_size, smaller_size) -> int:
        return math.ceil(larger_size / smaller_size)


class _AdaptiveConvNd(Nerve):
    
    def __init__(self,
        spatial_dims,
        kernel_size,
        dilation=1,
        transposed: bool = False,
    ) -> None:
        super().__init__()

        N = len(spatial_dims)
        self.transposed: bool = transposed
        self.spatial_dims: list = list(spatial_dims)
        self.default_dims: bool = self.spatial_dims == list(range(-N, 0, 1))
        self.num_dims = N
        self.adapted_to_base_shape = None
        self._readaptation_warning_flag = True
        self._ntuple = ntuple(N)
        self.kernel_size = self._ntuple(kernel_size)
        self.dilation = self._ntuple(dilation)
        self.stride: Tuple[int] = None
        self.padding = None
        self.output_padding = None
        self._reversed_padding_repeated_twice = None

    def _adapt_to_base_shape(self):
        # only execute the remaining code when base_shape does not match.
        base_shape = self.base_shape[self.spatial_dims]
        if base_shape == self.adapted_to_base_shape:
            return
        self.adapted_to_base_shape = base_shape

        if self._readaptation_warning_flag and self.stride is not None:
            self._readaptation_warning_flag = False
            warnings.warn("`nv.ConvNd` shape adaptation occured multiple times.")

        def convert_to_absolute(shape_spec: ShapeSpec):
            if shape_spec.relative:
                down_scales = Shape(shape_spec.relative, repeat_times=self.num_dims)
                abs_shape = []
                for base_size, down_scale in zip(base_shape, down_scales):
                    abs_size = ConvArithmetic.size(base_size, down_scale)
                    abs_shape.append(abs_size)
                return Shape(abs_shape)
            else:
                return shape_spec.absolute

        in_shape = self.input_links.dims[0].shape
        out_shape = self.target_dims[0].shape

        # populate self.stride
        # -- relative case
        if in_shape.relative and out_shape.relative:
            in_shape = Shape(in_shape.relative, repeat_times=self.num_dims)
            out_shape = Shape(out_shape.relative, repeat_times=self.num_dims)
            stride_tuple = []
            for in_size, out_size in zip(in_shape, out_shape):
                if self.transposed:
                    stride_size = in_size / out_size
                else:
                    stride_size = out_size / in_size
                try:
                    stride_tuple.append(expect_int(stride_size))
                except TypeError:
                    raise ValueError(
                        f"stride {stride_size} is not an integer. "
                        f"(in_size={in_size}, out_size={out_size}, transposed={self.transposed}, class={self.__class__.__name__})"
                    )
            self.stride = tuple(stride_tuple)
        # -- absolute case
        else:
            in_shape = convert_to_absolute(in_shape)
            out_shape = convert_to_absolute(out_shape)

            stride_tuple = []
            for in_size, out_size in zip(in_shape, out_shape):
                if self.transposed:
                    if out_size < in_size:
                        raise ValueError(
                            f"output_size should be larger than input_size in transposed convolution, but {out_size} < {in_size}."
                            f"(in_size={in_size}, out_size={out_size}, transposed={self.transposed}, class={self.__class__.__name__})"
                        )
                    stride_size = ConvArithmetic.stride(out_size, in_size)
                else:
                    if out_size > in_size:
                        raise ValueError(
                            f"output_size should be less than input_size in convolution, but {out_size} > {in_size}."
                            f"(in_size={in_size}, out_size={out_size}, transposed={self.transposed}, class={self.__class__.__name__})"
                        )
                    stride_size = ConvArithmetic.stride(in_size, out_size)
                try:
                    stride_tuple.append(expect_int(stride_size))
                except TypeError:
                    raise NeurlinkAssertionError(
                        f"assert isint(stride_size={stride_size})"
                    )
            self.stride = tuple(stride_tuple)

        # populate self.padding
        in_shape = convert_to_absolute(in_shape)
        out_shape = convert_to_absolute(out_shape)
        padding_tuple = []
        output_padding_tuple = []
        for in_size, out_size, k, d, s in zip(
            in_shape, out_shape, self.kernel_size, self.dilation, self.stride
        ):
            if self.transposed:
                p, a = ConvArithmetic.padding_transposed(in_size, out_size, k, d, s)
            else:
                p = ConvArithmetic.padding(in_size, out_size, k, d, s)
                a = None
            padding_tuple.append(p)
            output_padding_tuple.append(a)
        self.padding = tuple(padding_tuple)
        self.output_padding = tuple(output_padding_tuple)
        self._reversed_padding_repeated_twice = reverse_repeat_tuple(self.padding, 2)
    
    def __call__(self, x):
        self._adapt_to_base_shape()
        return self.forward(x)

    def forward(self, inputs):
        raise NotImplementedError()


class _ConvNd(_AdaptiveConvNd):
    def __init__(
        self,
        spatial_dims,
        kernel_size,
        dilation=1,
        groups=1,
        bias=False,
        norm=None,
        act=nn.ReLU,
        padding_mode: str = "zeros",
        transposed: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(spatial_dims, kernel_size, dilation, transposed)

        N = self.num_dims

        if transposed:
            if N == 1:
                self.functional_conv = F.conv_transpose1d
            elif N == 2:
                self.functional_conv = F.conv_transpose2d
            elif N == 3:
                self.functional_conv = F.conv_transpose3d
            else:
                raise NotImplementedError(f"nv.ConvNd(spatial_dims={spatial_dims}, transposed={transposed})")
        else:
            if N == 1:
                self.functional_conv = F.conv1d
            elif N == 2:
                self.functional_conv = F.conv2d
            elif N == 3:
                self.functional_conv = F.conv3d
            else:
                raise NotImplementedError(f"nv.ConvNd(spatial_dims={spatial_dims}, transposed={transposed})")

        in_channels = self.input_links.dims[0].channels
        out_channels = self.target_dims[0].channels
        self.padding_mode = padding_mode
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if len(kernel_size) != N:
            raise ValueError(f"kernel_size must be a tuple of length {N}.")

        factory_kwargs = {"device": device, "dtype": dtype}
        if transposed:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.norm = norm(out_channels)
        self.act = act(inplace=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def conv(self, x):
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = self._ntuple(0)
        if self.transposed:
            return self.functional_conv(
                x, self.weight, self.bias, self.stride, padding, self.output_padding, self.groups, self.dilation
            )
        else:
            return self.functional_conv(
                x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x