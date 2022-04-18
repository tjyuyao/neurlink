import neurlink.nerves as nv
import neurlink.nn as nn


class BasicBlock(nv.Nerve):
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
                (dim_out, nv.SkipConnect2d[[0, -1]](1, norm=norm)),
                (dim_out, act(inplace=True)),
            ]
        )


class Bottleneck(nv.Nerve):
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

        # fmt: off
        self.add(
            [
                ((hid_channels, dim_in.shape), nv.Conv2d(1, act=act, **conv_keywords)),
                ((hid_channels, dim_out.shape), nv.Conv2d(3, act=act, **conv_keywords)),
                ((out_channels, dim_out.shape), nv.Conv2d(1, act=nn.I, **conv_keywords)),
                ((out_channels, dim_out.shape), nv.SkipConnect2d[[0, -1]](1, norm=norm)),
                ((out_channels, dim_out.shape), act(inplace=True)),
            ]
        )
        # fmt: on


def _resnet(block, layers, expansion, num_classes):
    return nv.build(
        [
            ((3, 1), nv.Input()),
            ((64, 2), nv.Conv2d_ReLU_BN(7)),
            ((64, 4), nv.MaxPool2d(3)),
            [((64 * expansion, 4), block)] * layers[0],
            [((128 * expansion, 8), block)] * layers[1],
            [((256 * expansion, 16), block)] * layers[2],
            [((512 * expansion, 32), block)] * layers[3],
            ((512 * expansion, "(1, 1)"), nv.AvgPool2d()),
            ((num_classes, "(1, 1)"), nv.Conv2d(1)),
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
