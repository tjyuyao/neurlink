import torch
from neurlink.nerves.nerve import *


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
    test_base_shape()