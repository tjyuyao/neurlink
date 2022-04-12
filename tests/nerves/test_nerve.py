# import torch
# from neurlink.nerves.nerve import *

# def test_expand():
#     nndefs = list(expand([
#         ((3, 1), Input()),
#         ((6, 1), Nerve()),
#         [((7, 1), Nerve())] * 2,
#     ]))
    
#     assert nndefs[0][0] == (3, 1)
#     assert nndefs[1][0] == (6, 1)
#     assert nndefs[2][0] == (7, 1)
#     assert nndefs[3][0] == (7, 1)


# def test_base_size():

#     class AssumeBaseSize(Nerve):

#         def __init__(self, assumed) -> None:
#             super().__init__()
#             self.assumed = assumed
        
#         def forward(self):
#             assert self.assumed == self.base_size

#     net = Network([
#         ((3, 1), Input()),
#         ((6, 1), AssumeBaseSize((256, 512))),
#     ])

#     x = torch.rand((1, 3, 256, 512))
#     net(x)

