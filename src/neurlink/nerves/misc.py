from turtle import forward
from typing import Sequence
from .nerve import Nerve


class Identity(Nerve):
    def forward(self, x):
        if isinstance(x, Sequence) and len(x) == 1:
            return x[0]
        else:
            return x


class Add(Nerve):
    def forward(self, x):
        return x[0] + x[1]
