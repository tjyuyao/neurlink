from functools import partial
from typing import Type

import torch.nn as nn

Conv2d_1x1:Type[nn.Conv2d] = partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
Conv2d_3x3:Type[nn.Conv2d] = partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
