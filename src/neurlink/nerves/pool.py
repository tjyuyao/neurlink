import math
from turtle import forward
import warnings
from typing import Tuple, Type

import neurlink.nn as nn
import neurlink.nn.functional as F
import torch

from .common_types import NeurlinkAssertionError, size_any_t
from .nerve import Nerve, Shape, ShapeSpec
from .utils import expect_int, ntuple, specialize
from .conv import AdaptiveConvNd


class _MaxPoolNd(AdaptiveConvNd):
    def __init__(
        self,
        kernel_size: size_any_t=None,
        dilation=1,
        padding_mode="zeros",
        return_indices: bool = False,
        ceil_mode: bool = False,
        spatial_dims: size_any_t=None,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        if kernel_size is not None:
            self.adaptive_kernel = False
            self.func = {
                False: {
                    1: F.max_pool1d,
                    2: F.max_pool2d,
                    3: F.max_pool3d,
                },
                True: {
                    1: F.max_pool1d_with_indices,
                    2: F.max_pool2d_with_indices,
                    3: F.max_pool3d_with_indices,
                },
            }[return_indices][self.num_dims]
        else:
            self.adaptive_kernel = True
            self.func = {
                False: {
                    1: F.adaptive_max_pool1d,
                    2: F.adaptive_max_pool2d,
                    3: F.adaptive_max_pool3d,
                },
                True: {
                    1: F.adaptive_max_pool1d_with_indices,
                    2: F.adaptive_max_pool2d_with_indices,
                    3: F.adaptive_max_pool3d_with_indices,
                },
            }[return_indices][self.num_dims]


    def forward(self, x):
        if self.adaptive_kernel:
            output = self.func(
                input=x,
                output_size=None,
                return_indices=self.return_indices,
            )
        else:
            output = self.func(
                input=x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )
        if self.return_indices:
            return list(output)  # tensor, indices
        else:
            return output


MaxPool1d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-1,))
MaxPool2d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-2, -1,))
MaxPool3d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-3, -2, -1,))