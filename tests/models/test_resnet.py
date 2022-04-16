import pytest
import torch
import neurlink.nerves as nv


def test_conv2d_same():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((64, 2), nv.Conv2d_ReLU_BN(7)),
        ((64, 4), nv.MaxPool2d(3)),
    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x)
    import ice
    ice.print(x)


if __name__ == "__main__":
    test_conv2d_same()