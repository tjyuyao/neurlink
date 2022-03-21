import torch
import neurlink

model = neurlink.resnet18()
x = torch.randn((2, 3, 224, 224))
out = model(x)
for y in out:
    print(y.shape)