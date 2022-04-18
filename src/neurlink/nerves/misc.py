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

    def forward(self, inputs):
        if self.size_cached is None:
            self.size_cached = self.output_shapes[0]
        return F.interpolate(inputs[0], self.size_cached, mode=self.mode)


class Identity(Nerve):
    def forward(self, x):
        return x


class Add(Nerve):
    def forward(self, x):
        return x[0] + x[1]
