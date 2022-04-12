import torch
import pytest
from neurlink.nerves.nerve import *

def test_getmeta_default():
    x = type("X", (object, ), {})
    assert NEURLINK_META_NOT_FOUND is getmeta(x, "dummy_key")

def test_selectors():
    class Dummy(Nerve):

        def __init__(self, assumed) -> None:
            super().__init__()
            input_links = self.input_links
            if not isinstance(input_links, list):
                input_links = [input_links]
            for link, a in zip(input_links, assumed):
                assert link[:-1] == a

    net = build([
        ((1, 1), Input()),
        ((2, 1), Input()),
        ((3, 1), Input["x3"]()),
        ((4, 1), Input()),
        ((5, 1), Input["x5"]()),
        ((0, 1), Dummy([((5, 1),)])),  # select last tensor by default
        ((0, 1), Dummy[-2]([((5, 1),)])),
        ((0, 1), Dummy[2:4]([((3, 1),), ((4, 1),)])),
        ((0, 1), Dummy["x3"]([((3, 1),),])),
        ((0, 1), Dummy["x3":"x5"]([((3, 1),), ((4, 1),)])),
    ])

    x = [
        torch.rand((1, 1, 256, 512)),
        torch.rand((2, 1, 256, 512)),
        torch.rand((3, 1, 256, 512)),
        torch.rand((4, 1, 256, 512)),
        torch.rand((5, 1, 256, 512)),
    ]
    net(x)

def test_input_oob():
    # expecting 2 Inputs
    net = build([
        ((1, 1), Input()),
        ((2, 1), Input()),
    ])

    # but providing only 1
    x = torch.rand((1, 1, 256, 512))

    # would cause NotEnoughInputError
    with pytest.raises(NotEnoughInputError):
        net(x)

def test_base_shape():

    class AssumeBaseShape(Nerve):

        def __init__(self, assumed) -> None:
            super().__init__()
            self.assumed = assumed
        
        def forward(self, _):
            assert self.assumed == self.base_shape

    net = build([
        ((3, 1), Input()),
        ((6, 1), AssumeBaseShape((1, 3, 256, 512))),
    ])

    x = torch.rand((1, 3, 256, 512))
    net(x)

if __name__ == "__main__":
    test_selectors()