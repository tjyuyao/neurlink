import math
import warnings
from typing import Tuple

import neurlink.nn as nn
import neurlink.nn.functional as F
import torch

from .common_types import NeurlinkAssertionError, size_any_t
from .nerve import Nerve, Shape, ShapeSpec
from .utils import expect_int, ntuple, specialize


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
        sum_p1_p2 = (d * k + (o - 1) * s - d - i + 1)
        p2 = sum_p1_p2 // 2
        p1 = sum_p1_p2 - p2
        return (p1, p2)

    @staticmethod
    def padding_transposed(it, ot, k, d, s) -> int:
        p1, _ = ConvArithmetic.padding(ot, it, k, d, s)
        a = ot - ((it - 1) * s - 2*p1 + d * (k - 1) + 1)
        return (p1, p1), a

    @staticmethod
    def stride(larger_size, smaller_size) -> int:
        return math.ceil(larger_size / smaller_size)


class _AdaptiveConvNd(Nerve):
    def __init__(
        self,
        spatial_dims: size_any_t,
        kernel_size: size_any_t,
        dilation=1,
        transposed: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(spatial_dims, int):  # 1d conv
            spatial_dims = (spatial_dims,)

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
        self._functional_padding = None
        self._reversed_padding = None
        self._output_padding = None

        if not self.default_dims:
            raise NeurlinkAssertionError(
                "Arbitrary spatial_dims currently not implemented, "
                "I would appreciate it if you submit an issue describing your use case or implementation advice."
            )

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
        in_shape = self.input_links.dims[0].shape
        out_shape = self.target_dims[0].shape
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
        padding_tuple = tuple(padding_tuple)
        self._output_padding = tuple(output_padding_tuple)
        self._reversed_padding = tuple(x for p in reversed(padding_tuple) for x in p)
        self._seperate_pad = False
        if self.padding_mode == "zeros":
            for p1, p2 in padding_tuple:
                if p1 != p2:
                    self._seperate_pad = True
                    self.padding_mode = "constant"
        else:
            self._seperate_pad = True
        if self._seperate_pad:
            self._functional_padding = self._ntuple(0)
        else:
            self._functional_padding = tuple(p for p, _ in padding_tuple)

    def __call__(self, x):
        self._adapt_to_base_shape()
        return self.forward(x)

    def forward(self, inputs):
        raise NotImplementedError()


class _ConvNd(_AdaptiveConvNd):
    def __init__(
        self,
        kernel_size,
        dilation=1,
        groups=1,
        bias=False,
        norm=None,
        act=nn.ReLU,
        norm_after_act=True,
        padding_mode: str = "zeros",
        transposed: bool = False,
        device=None,
        dtype=None,
        spatial_dims=None,
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
                raise NotImplementedError(
                    f"nv.ConvNd(spatial_dims={spatial_dims}, transposed={transposed})"
                )
        else:
            if N == 1:
                self.functional_conv = F.conv1d
            elif N == 2:
                self.functional_conv = F.conv2d
            elif N == 3:
                self.functional_conv = F.conv3d
            else:
                raise NotImplementedError(
                    f"nv.ConvNd(spatial_dims={spatial_dims}, transposed={transposed})"
                )

        in_channels = self.input_links.dims[0].channels
        out_channels = self.target_dims[0].channels
        kernel_size = self.kernel_size
        self.padding_mode = padding_mode
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError(f"{self.__class__.__name__}(in_channels={in_channels}, groups={groups}): in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError(f"{self.__class__.__name__}(out_channels={out_channels}, groups={groups}): out_channels must be divisible by groups")
        if len(kernel_size) != N:
            raise ValueError(f"{self.__class__.__name__}(kernel_size={kernel_size}): kernel_size must be a tuple of length {N}.")

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

        if norm_after_act:
            self.postproc = nn.Sequential(
                act(inplace=True),
                norm(out_channels),
            )
        else:
            self.postproc = nn.Sequential(
                norm(out_channels),
                act(inplace=True),
            )

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
        if self._seperate_pad:
            x = F.pad(x, self._reversed_padding, mode=self.padding_mode)

        if self.transposed:
            return self.functional_conv(
                x,
                self.weight,
                self.bias,
                self.stride,
                self._functional_padding,
                self._output_padding,
                self.groups,
                self.dilation,
            )
        else:
            return self.functional_conv(
                x,
                self.weight,
                self.bias,
                self.stride,
                self._functional_padding,
                self.dilation,
                self.groups,
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.postproc(x)
        return x


# fmt: off

Conv1d = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.Identity, act=nn.Identity)
Conv1dReLU = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.Identity, act=nn.ReLU)
Conv1dBNReLU = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.BatchNorm1d, act=nn.ReLU, norm_after_act=False)
Conv1dReLUBN = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.BatchNorm1d, act=nn.ReLU, norm_after_act=True)

Conv2d = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.Identity, act=nn.Identity)
Conv2dReLU = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.Identity, act=nn.ReLU)
Conv2dBNReLU = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=False)
Conv2dReLUBN = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=True)

Conv3d = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.Identity, act=nn.Identity)
Conv3dReLU = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.Identity, act=nn.ReLU)
Conv3dBNReLU = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=False)
Conv3dReLUBN = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=True)

ConvTransposed1d = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.Identity, act=nn.Identity, transposed=True)
ConvTransposed1dReLU = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.Identity, act=nn.ReLU, transposed=True)
ConvTransposed1dBNReLU = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.BatchNorm1d, act=nn.ReLU, norm_after_act=False, transposed=True)
ConvTransposed1dReLUBN = specialize(_ConvNd, spatial_dims=(-1,), norm=nn.BatchNorm1d, act=nn.ReLU, norm_after_act=True, transposed=True)

ConvTransposed2d = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.Identity, act=nn.Identity, transposed=True)
ConvTransposed2dReLU = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.Identity, act=nn.ReLU, transposed=True)
ConvTransposed2dBNReLU = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=False, transposed=True)
ConvTransposed2dReLUBN = specialize(_ConvNd, spatial_dims=(-2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=True, transposed=True)

ConvTransposed3d = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.Identity, act=nn.Identity, transposed=True)
ConvTransposed3dReLU = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.Identity, act=nn.ReLU, transposed=True)
ConvTransposed3dBNReLU = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=False, transposed=True)
ConvTransposed3dReLUBN = specialize(_ConvNd, spatial_dims=(-3, -2, -1,), norm=nn.BatchNorm2d, act=nn.ReLU, norm_after_act=True, transposed=True)