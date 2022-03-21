# NEURLINK

A compact grammar for neural network definition based on PyTorch.

## Basics

- Nodes:
    - `(c, s, o)`: tensor (channel_size, downsample_scale, operation),
    - `((c1, s1), (c2, s2), ..., o)` or `((c, s, o),)*n`: parallel tensors,
    - `[(c, s, o)]*n`: sequential tensors,
- Ordered nodes form a graph. Each node has an operation (also known as transform function) that converts one or more previous tensors to (a) new tensor(s). Defining new operations requires some specific arguments and a decorator.

## Example ([resnet](neurlink/models/resnet.py))

```py
def resnet50(num_classes: int = 1000, **block_keywords):
    block = BottleNeck(**block_keywords)
    expansion = 4
    return build(
        [
            (3, 1, None),
            (64, 2, ConvLayer(7)),
            (64, 4, MaxPool2d(3)),
            [(64  * expansion, 4, block)] * 3,
            [(128 * expansion, 8, block)] * 4,
            [(256 * expansion, 16, block)] * 6,
            [(512 * expansion, 32, block)] * 3,
            (512 * expansion, "(1, 1)", AvgPool2d()),
            (num_classes, None, ConvLayer(1, norm=nn.Identity, act=nn.Identity)),
        ]
    )
```