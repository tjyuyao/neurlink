from turtle import forward
from typing import Sequence
from .nerve import Nerve
import neurlink.nn.functional as F


class Interpolate(Nerve):

    def __init__(self, mode) -> None:
        super().__init__()
        self.size_cached = None

        # mode (str): algorithm used for upsampling:
        #     ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        #     ``'trilinear'`` | ``'area'`` | ``'nearest-exact'``. Default: ``'nearest'``
        self.mode = mode

        assert self.input_links[0].dims[0].channels == self.target_dims[0].channels

    def forward(self, inputs):
        if self.size_cached is None:
            self.size_cached = tuple(self.output_shapes[0][2:])
        return F.interpolate(inputs[0], self.size_cached, mode=self.mode, align_corners=False)


class Identity(Nerve):
    def forward(self, x):
        return x


class Add(Nerve):
    def forward(self, x):
        return x[0] + x[1]
