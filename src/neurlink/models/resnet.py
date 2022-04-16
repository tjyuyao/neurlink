import neurlink.nerves as nv
from neurlink.nerves.nerve import DimSpec
import neurlink.nn as nn
import neurlink.nn.functional as F
import torch


class BasicResBlock(nv.Nerve):
    def __init__(
        self,
        width=1.0,
        groups=1,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        norm_after_act=True,
        expansion=1,
    ):
        super().__init__()

        assert len(self.input_links) == 1
        assert expansion == 1, "BasicResBlock only supports expansion=1"

        dim_in = self.input_links[0].dims[0]
        dim_out = self.target_dims[0]
        conv_keywords = dict(norm=norm, act=act, norm_after_act=norm_after_act, groups=groups)

        hid_channels = int(dim_out.channels * width)

        input = 0

        if dim_in != dim_out:
            skip = self.add((dim_out, nv.Conv2d[input, "skip"](1, norm=norm, groups=groups)))
        else:
            skip = input
        
        conv1 = self.add(((hid_channels, dim_out.shape), nv.Conv2d[input, "conv1"](3, **conv_keywords)))
        conv2 = self.add((dim_out, nv.Conv2d[conv1, "conv2"](3, **conv_keywords)))
        self.add((dim_out, nv.Add[[conv2, skip]]()))
        self.act = act(inplace=True)

    def forward(self, inputs):
        x = super().forward(inputs)
        return self.act(x[-1])