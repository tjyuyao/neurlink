import torch
import neurlink.nerves as nv

def test_conv2d():
    net = nv.Network([
        ((3, 1), nv.Input()),
        ((6, 1), nv.Conv2d(3)),
        ((7, 1), nv.Conv2d(3, dilation=3)),

    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x)
    assert x[0].shape == (2, 3, 224, 256)
    assert x[1].shape == (2, 6, 224, 256)
    print(x[2].shape)

