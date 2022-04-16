from typing import Type

import neurlink.nn.functional as F

from .common_types import size_any_t
from .conv import AdaptiveConvNd
from .utils import specialize


class _MaxPoolNd(AdaptiveConvNd):
    def __init__(
        self,
        kernel_size: size_any_t = None,
        dilation=1,
        padding_mode="zeros",
        return_indices: bool = False,
        spatial_dims: size_any_t = None,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size or 1,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.return_indices = return_indices
        self.ceil_mode = False

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
                output_size=self.output_shapes[0],
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


# fmt: off
MaxPool1d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-1,))
MaxPool2d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-2, -1,))
MaxPool3d: Type[_MaxPoolNd] = specialize(_MaxPoolNd, spatial_dims=(-3, -2, -1,))
# fmt: on


class _AvgPoolNd(AdaptiveConvNd):
    def __init__(
        self,
        kernel_size: size_any_t = None,
        padding_mode="zeros",
        count_include_pad=True,
        spatial_dims: size_any_t = None,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            kernel_size=kernel_size or 1,
            padding_mode=padding_mode,
        )

        self.count_include_pad = count_include_pad
        self.ceil_mode = False

        if kernel_size is not None:
            self.adaptive_kernel = False
            self.func = {
                1: F.avg_pool1d,
                2: F.avg_pool2d,
                3: F.avg_pool3d,
            }[self.num_dims]
        else:
            self.adaptive_kernel = True
            self.func = {
                1: F.adaptive_avg_pool1d,
                2: F.adaptive_avg_pool2d,
                3: F.adaptive_avg_pool3d,
            }[self.num_dims]

    def forward(self, x):
        if self.adaptive_kernel:
            output = self.func(
                input=x,
                output_size=self.output_shapes[0],
            )
        else:
            output = self.func(
                input=x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        return output


# fmt: off
AvgPool1d: Type[_AvgPoolNd] = specialize(_AvgPoolNd, spatial_dims=(-1,))
AvgPool2d: Type[_AvgPoolNd] = specialize(_AvgPoolNd, spatial_dims=(-2, -1,))
AvgPool3d: Type[_AvgPoolNd] = specialize(_AvgPoolNd, spatial_dims=(-3, -2, -1,))
