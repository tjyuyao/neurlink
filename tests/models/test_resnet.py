from numpy import block
import pytest
import torch
import neurlink.nerves as nv
from neurlink.models.resnet import *


def test_conv2d_same(expansion=1):
    block = BasicResBlock(3, expansion=1)
    net = nv.build([
        ((3, 1), nv.Input()),
        ((64, 2), nv.Conv2d_ReLU_BN(7)),
        ((64, 4), nv.MaxPool2d(3)),
        [((64  * expansion, 4), block)] * 3,
        [((128  * expansion, 8), block)] * 4,
        [((256  * expansion, 16), block)] * 6,
        [((512  * expansion, 32), block)] * 3,
    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x)
    import ice
    ice.print(x)


if __name__ == "__main__":
    test_conv2d_same()