from neurlink.nerves.common_types import NNDefParserError
import pytest
import torch
import neurlink.nerves as nv
from neurlink.nerves.conv import *


def test_conv2d_same():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((6, 1), Conv2d(3)),
        ((8, 1), Conv2d(4, dilation=7)),
        ((8, 1), Conv2d(5, groups=2)),

    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x, output_intermediate=True)
    assert x[0].shape == (2, 3, 224, 256)
    assert x[1].shape == (2, 6, 224, 256)
    assert x[2].shape == (2, 8, 224, 256)
    assert x[3].shape == (2, 8, 224, 256)

def test_conv2d_downsizing_even():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((6, 2), Conv2d(3)),
        ((8, 4), Conv2d(4, dilation=7)),
        ((8, 8), Conv2d(5, groups=2)),
        ((8, 16), Conv2d(2, dilation=3)),
        ((8, 32), Conv2d(7, dilation=2)),
    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x, output_intermediate=True)
    assert x[0].shape == (2, 3, 224, 256)
    assert x[1].shape == (2, 6, 112, 128)
    assert x[2].shape == (2, 8, 56, 64)
    assert x[3].shape == (2, 8, 28, 32)
    assert x[4].shape == (2, 8, 14, 16)
    assert x[5].shape == (2, 8, 7, 8)

def test_conv2d_downsizing_odd():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((6, 2), Conv2d(3)),
        ((8, 4), Conv2d(4, dilation=7)),
        ((8, 8), Conv2d(5, groups=2)),
        ((8, 16), Conv2d(2, dilation=3, padding_mode="reflect")),
        ((8, 32), Conv2d(7, dilation=2)),
    ])
    x = torch.randn((2, 3, 224+1, 256+1))
    x = net(x, output_intermediate=True)
    assert x[0].shape == (2, 3, 224+1, 256+1)
    assert x[1].shape == (2, 6, 112+1, 128+1)
    assert x[2].shape == (2, 8, 56+1, 64+1)
    assert x[3].shape == (2, 8, 28+1, 32+1)
    assert x[4].shape == (2, 8, 14+1, 16+1)
    assert x[5].shape == (2, 8, 7+1, 8+1)

def test_conv2d_transposed_even():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((6, 2), Conv2d(3)),
        ((8, 1), ConvTransposed2d(3)),
    ])
    x = torch.randn((2, 3, 224, 256))
    x = net(x, output_intermediate=True)
    assert x[0].shape == (2, 3, 224, 256)
    assert x[1].shape == (2, 6, 112, 128)
    assert x[2].shape == (2, 8, 224, 256)

def test_conv2d_transposed_odd():
    net = nv.build([
        ((3, 1), nv.Input()),
        ((6, 2), Conv2d(3)),
        ((8, 1), ConvTransposed2d(3)),
    ])
    x = torch.randn((2, 3, 224+1, 256+1))
    x = net(x, output_intermediate=True)
    assert x[0].shape == (2, 3, 224+1, 256+1)
    assert x[1].shape == (2, 6, 112+1, 128+1)
    assert x[2].shape == (2, 8, 224+1, 256+1)

def test_conv2d_transposed_nonzero_padding():
    with pytest.raises(ValueError):
        nv.build([
            ((3, 1), nv.Input()),
            ((6, 2), Conv2d(3)),
            ((8, 1), ConvTransposed2d(3, padding_mode="reflect")),
        ])

if __name__ == "__main__":
    test_conv2d_same()