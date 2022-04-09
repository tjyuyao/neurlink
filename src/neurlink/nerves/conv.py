import neurlink.nn as nn

from .nerve import Nerve

class Conv2d(Nerve):
    def __init__(
        self,
        kernel_size,
        dilation=1,
        groups=1,
        bias=False,
        norm=nn.BatchNorm2d,
        act=nn.ReLU
    ):
        super().__init__()

        in_channels, in_scale = self.input_links[:2]
        out_channels, out_scale = self.target_shapes[0]
        if out_scale:
            stride = out_scale // in_scale
            assert stride >= 1
        else:
            stride = 1
        padding = kernel_size * dilation // 2

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.norm = norm(out_channels)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


