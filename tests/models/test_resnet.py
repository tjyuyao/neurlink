import torch
from neurlink.models.resnet import *


def test_resnets_cuda():
    for net_builder in [resnet18, resnet50, resnext50_32x4d, wide_resnet50_2]:
        net = net_builder(num_classes=10).cuda()
        x = torch.randn((2, 3, 64, 64)).cuda()
        x = net(x)
        assert x.shape == (2, 10, 1, 1)


def test_speed():
    net = resnet50().cuda()
    torch.cuda.synchronize()
    import tqdm
    for _ in tqdm.trange(100):
        x = torch.randn((2, 3, 224, 224)).cuda()
        x = net(x)

if __name__ == "__main__":
    test_speed()