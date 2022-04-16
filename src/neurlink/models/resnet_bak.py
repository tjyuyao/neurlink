from typing import Optional
from .. import nn
from ..nn import functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        dilation=1,
        groups=1,
        bias=False,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        *,
        shape,
        input_shapes,
        input_index=-1,
        name=None,
    ):
        super().__init__()

        in_channels, in_scale = input_shapes[input_index][:2]
        out_channels, out_scale = shape[:2]
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
        self.input_index = input_index
        self.name = name

    def forward(self, cache):
        x = cache[self.input_index]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size=None,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        *,
        shape,
        input_shapes,
        input_index=-1,
        name=None,
    ):
        super().__init__()

        in_channels, in_scale = input_shapes[input_index][:2]
        out_channels, out_scale = shape[:2]

        assert out_channels == in_channels

        if isinstance(out_scale, (int)):
            stride = out_scale // in_scale
            assert stride >= 1
            padding = kernel_size * dilation // 2

            self.pool = nn.MaxPool2d(
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
        elif isinstance(out_scale, (str)):
            self.pool = nn.AdaptiveMaxPool2d(
                eval(out_scale), return_indices=return_indices
            )

        self.indices = None
        self.return_indices = return_indices
        self.input_index = input_index
        self.name = name

    def forward(self, cache):
        x = cache[self.input_index]
        if self.return_indices:
            x, self.indices = self.pool(x)
        else:
            x = self.pool(x)
        return x


class AvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size=None,
        ceil_mode=False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        shape,
        input_shapes,
        input_index=-1,
        name=None,
    ):
        super().__init__()

        in_channels, in_scale = input_shapes[input_index][:2]
        out_channels, out_scale = shape[:2]
        assert out_channels == in_channels

        if isinstance(out_scale, (int)):
            stride = out_scale // in_scale
            assert stride >= 1
            padding = kernel_size // 2

            self.pool = nn.AvgPool2d(
                kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
        elif isinstance(out_scale, (str)):
            self.pool = nn.AdaptiveAvgPool2d(eval(out_scale))

        self.input_index = input_index
        self.name = name

    def forward(self, cache):
        x = cache[self.input_index]
        x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        width=1.0,
        groups=1,
        droppath=0.0,
        dropout=0.0,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        *,
        shape,
        input_shapes,
        input_index=-1,
        name=None,
    ):
        super().__init__()

        in_channels, in_scale = input_shapes[input_index][:2]
        out_channels, out_scale = shape[:2]
        stride = out_scale // in_scale
        assert stride >= 1

        hid_channels = int(out_channels * width)

        self.shortcut = []
        if stride != 1 or in_channels != out_channels:
            self.shortcut.append(
                nn.Conv2d_1x1(in_channels, out_channels, stride=stride, groups=groups)
            )
            self.shortcut.append(norm(out_channels))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            nn.Conv2d_3x3(in_channels, hid_channels, stride=stride, groups=groups),
            norm(hid_channels),
            act(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d_3x3(hid_channels, out_channels),
            norm(out_channels),
        )

        self.act = act(inplace=True)
        self.sd = nn.DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.do = dropout
        self.input_index = input_index
        self.name = name

    def forward(self, cache):
        x = cache[self.input_index]
        skip = self.shortcut(x)
        x = self.conv1(x)
        if self.do:
            x = F.dropout(x, p=self.do, training=self.training, inplace=True)
        x = self.conv2(x)
        x = self.sd(x) + skip
        x = self.act(x)

        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        width=1.0,
        expansion=4.0,
        groups=1,
        droppath=0.0,
        dropout=0.0,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        *,
        shape,
        input_shapes,
        input_index=-1,
        name=None,
    ):
        super().__init__()

        in_channels, in_scale = input_shapes[input_index][:2]
        out_channels, out_scale = shape[:2]
        stride = out_scale // in_scale
        assert stride >= 1

        hid_channels = int(out_channels * width / expansion)

        self.shortcut = []
        if stride != 1 or in_channels != out_channels:
            self.shortcut.append(
                nn.Conv2d_1x1(in_channels, out_channels, stride=stride)
            )
            self.shortcut.append(norm(out_channels))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            nn.Conv2d_1x1(in_channels, hid_channels),
            norm(hid_channels),
            act(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d_3x3(hid_channels, hid_channels, stride=stride, groups=groups),
            norm(hid_channels),
            act(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d_1x1(hid_channels, out_channels),
            norm(out_channels),
        )

        self.act = act(inplace=True)
        self.sd = nn.DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.do = dropout

        self.input_index = input_index
        self.name = name

    def forward(self, cache):
        x = cache[self.input_index]
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        if self.do:
            x = F.dropout(x, p=self.do, training=self.training, inplace=True)
        x = self.conv3(x)

        x = self.sd(x) + skip
        x = self.act(x)

        return x


def _resnet(block, layers, expansion, num_classes):
    return build(
        [
            (3, 1, None),
            (64, 2, ConvLayer(7)),
            (64, 4, MaxPool2d(3)),
            [(64 * expansion, 4, block)] * layers[0],
            [(128 * expansion, 8, block)] * layers[1],
            [(256 * expansion, 16, block)] * layers[2],
            [(512 * expansion, 32, block)] * layers[3],
            (512 * expansion, "(1, 1)", AvgPool2d()),
            (num_classes, None, ConvLayer(1, norm=nn.Identity, act=nn.Identity)),
        ]
    )


def resnet18(num_classes: int = 1000, **block_keywords):
    return _resnet(
        block=BasicBlock(**block_keywords),
        layers=[2, 2, 2, 2],
        expansion=1,
        num_classes=num_classes,
    )


def resnet34(num_classes: int = 1000, **block_keywords):
    return _resnet(
        block=BasicBlock(**block_keywords),
        layers=[3, 4, 6, 3],
        expansion=1,
        num_classes=num_classes,
    )


def resnet50(num_classes: int = 1000, **block_keywords):
    return _resnet(
        block=Bottleneck(**block_keywords),
        layers=[3, 4, 6, 3],
        expansion=4,
        num_classes=num_classes,
    )


def resnet101(num_classes: int = 1000, **block_keywords):
    return _resnet(
        block=Bottleneck(**block_keywords),
        layers=[3, 4, 23, 3],
        expansion=4,
        num_classes=num_classes,
    )


def resnet152(num_classes: int = 1000, **block_keywords):
    return _resnet(
        block=Bottleneck(**block_keywords),
        layers=[3, 8, 36, 3],
        expansion=4,
        num_classes=num_classes,
    )


def resnext50_32x4d(num_classes: int = 1000, **block_keywords):
    return resnet50(num_classes, groups=32, width=4 * 32 / 64, **block_keywords)


def resnext101_32x8d(num_classes: int = 1000, **block_keywords):
    return resnet101(num_classes, groups=32, width=8 * 32 / 64, **block_keywords)


def wide_resnet50_2(num_classes: int = 1000, **block_keywords):
    return resnet50(num_classes, width=2, **block_keywords)


def wide_resnet101_2(num_classes: int = 1000, **block_keywords):
    return resnet101(num_classes, width=2, **block_keywords)
