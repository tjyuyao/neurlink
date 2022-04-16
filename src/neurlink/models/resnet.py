import neurlink.nerves as nv
from neurlink.nerves.nerve import DimSpec
import neurlink.nn as nn
import neurlink.nn.functional as F
import torch


class SkipConnect(nv.Nerve):
    """
        SkipConnect[[from, to]](...)
    """

    def __init__(
        self,
        kernel_size=1,
        norm=nn.BatchNorm2d,
        act=nn.Identity,
        **conv_keywords,
    ):
        super().__init__()
        dim_from = self.input_links[0].dims[0]
        dim_to = self.input_links[1].dims[0]
        dim_out = self.target_dims[0]

        assert dim_out == dim_to        

        if dim_from != dim_to:
            self.add(
                (dim_to, nv.Conv2d[0](kernel_size, norm=norm, act=act, **conv_keywords))
            )
        
    def forward(self, inputs, output_intermediate=False):
        cache = super().forward(inputs, output_list=True)
        cache.append(cache[-1] + cache[-2])
        if output_intermediate:
            return cache
        else:
            return cache[-1]


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

        dim_out = self.target_dims[0]
        conv_keywords = dict(norm=norm, norm_after_act=norm_after_act, groups=groups)

        hid_dims = (int(dim_out.channels * width), dim_out.shape)

        self.add(
            [
                (hid_dims, nv.Conv2d(3, act=act, **conv_keywords)),
                (dim_out, nv.Conv2d(3, act=nn.I, **conv_keywords)),
                (dim_out, SkipConnect[[0, -1]](1, norm=norm)),
                (dim_out, act(inplace=True)),
            ]
        )


class BottleneckResBlock(nv.Nerve):
    def __init__(
        self,
        width=1.0,
        groups=1,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        norm_after_act=True,
        expansion=4,
    ):
        super().__init__()

        assert len(self.input_links) == 1

        dim_in = self.input_links[0].dims[0]
        dim_out = self.target_dims[0]
        conv_keywords = dict(norm=norm, norm_after_act=norm_after_act, groups=groups)

        out_channels = dim_out.channels
        hid_channels = int(dim_out.channels * width / expansion)

        self.add(
            [
                ((hid_channels, dim_in.shape ), nv.Conv2d(1, act=act, **conv_keywords)),
                ((hid_channels, dim_out.shape), nv.Conv2d(3, act=act, **conv_keywords)),
                ((out_channels, dim_out.shape), nv.Conv2d(1, act=nn.I, **conv_keywords)),
                ((out_channels, dim_out.shape), SkipConnect[[0, -1]](1, norm=norm)),
                ((out_channels, dim_out.shape), act(inplace=True)),
            ]
        )