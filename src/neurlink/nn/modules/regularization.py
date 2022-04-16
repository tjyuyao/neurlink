import torch.nn as nn
from ..functional.regularization import drop_path


class DropPath(nn.Module):
    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)
