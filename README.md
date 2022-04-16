# NEURLINK

A compact grammar for neural network definition based on PyTorch.

## Basics

Neurlink asks for a sequence of `nndef` specification lines to define a flexibly connected neural network. 
Each line of `nndef` syntax follows the pattern `((C1, S2), ..., (Cn, Sn), NerveClass[input_selector, tag](...))` that says a `nerve` takes inputs determined by `input_selector`
and output `n` tensors, each of which has channel size `C` and spatial shape `S`.

### input selector

Every `nndef` node can select its input from the set of all previous lines. By default, `input_selector` can be omitted and means outputs of previous `nndef`; it can also be a integer index, a `slice`, or a `list`, so that one can specify multiple input nodes. The implementation of each layer is a subclass of `Nerve` and can get informations of the input specifications by attribute `self.input_links`.

`tag` can be optionally specified to allow you refer to a `nndef` specification by a string alias.

### output dimensions

Output dimensions specify channels and spatial shapes seperately. The spatial shape written as a (tuple of) integer/float number is a **relative** down sampling ratio of the `base_shape` (specified by `nv.Input`). It can also be written as a string so that it evaluates to absolute shape. The dimensions are written out to give a straightforward impression of the computation flow of the entire network, and shape transformation is done automatically so that you don't bother to derive how many stride or padding you might want.

## Example ([resnet](src/neurlink/models/resnet.py))

```py
def resnet50(num_classes: int = 1000, **block_keywords):
    block = BottleNeck(**block_keywords, expansion=4)
    expansion = 4
    return build(
        [
            ((3, 1), nv.Input()),
            ((64, 2), Conv2d_ReLU_BN(7)),  # 7x7 conv, stride 2
            ((64, 4), MaxPool2d(3)),  # 3x3 maxpool, stride 2
            [((64  * expansion, 4), block)] * 3,  # 3 layers of residual blocks, w/o downsampling
            [((128 * expansion, 8), block)] * 4,  # 4 layers of residual blocks, downsampling (x2) happens at the first block
            [((256 * expansion, 16), block)] * 6,  # 256 * expansion is the actual output channel, the bottleneck shape is interpreted inside of the block.
            [((512 * expansion, 32), block)] * 3,  # finally downsamped to 1/32 of origin, you may have noted that you can easily figure out the global downsample ratio.
            ((512 * expansion, "(1, 1)"), AvgPool2d()),  # a average pooling layer downsamples to an absolute shape.
            ((num_classes, "(1, 1)"), Conv2d(1)),  # final linear layer.
        ]
    )
```

```py
# in tests/test_build_models.py
import torch
import neurlink

model = neurlink.resnet18()
x = torch.randn((2, 3, 224, 224))
out = model(x, output_intermediate=True)
for y in out:
    print(y.shape)
```

```bash
$ python tests/test_build_models.py

torch.Size([2, 3, 224, 224])
torch.Size([2, 64, 112, 112])
torch.Size([2, 64, 56, 56])
torch.Size([2, 64, 56, 56])
torch.Size([2, 64, 56, 56])
torch.Size([2, 128, 28, 28])
torch.Size([2, 128, 28, 28])
torch.Size([2, 256, 14, 14])
torch.Size([2, 256, 14, 14])
torch.Size([2, 512, 7, 7])
torch.Size([2, 512, 7, 7])
torch.Size([2, 512, 1, 1])
torch.Size([2, 1000, 1, 1])
```
