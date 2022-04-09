from ast import arguments
import collections
from copy import copy
from typing import overload
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, wraps


def transform_function(cls):

    @wraps(cls)
    def factory(*args, **kwds):
        return partial(cls, *args, **kwds)

    return factory


def expand(nndefs):
    if isinstance(nndefs, list):
        for i in nndefs:
            yield from expand(i)
    else:
        yield nndefs


class IndexTypeError(Exception): ...


class Index:

    def __init__(self, i) -> None:
        if isinstance(i, Index):
            self.base = i.base
            self.offset = i.offset
        elif isinstance(i, int):
            self.base = "0"
            self.offset = i
        elif isinstance(i, str):
            self.base = i
            self.offset = i
        elif isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], str) and isinstance(i[1], int):
            self.base = i[0]
            self.offset = i[1]
        else:
            raise IndexTypeError()
    
    def __str__(self) -> str:
        if self.base == "0":
            return f"neurlink.Index({self.offset})"
        else:
            return f"neurlink.Index({self.base}, {self.offset})"
        
    def __add__(self, other: "Index"):
        if isinstance(other, int):
            return Index((self.base, self.offset + other))
        elif isinstance(other, Index) and other.base == "0":
            return Index((self.base, self.offset + other.offset))
        else:
            raise IndexTypeError()


class IndexRange:

    def __init__(self, start:Index = -1, stop:Index = None):
        self.start = Index(start)
        self.stop = (self.start + 1) if stop is None else Index(stop)


class Network(nn.Module):

    def __init__(self, nndefs) -> None:
        super().__init__()
        nodeseq = []
        modules = {}
        name2idx = {"0": 0}
        for idx, nndef in enumerate(expand(nndefs)):
            if isinstance(nndef[-1], partial):
                p: partial = nndef[-1]
                module = p(shape=nndef, input_shapes=copy(nodeseq))
                module.__pstr__ = stringfy_partial(p)
                nndef = nndef[:-1] + (module,)
                if hasattr(module, "name") and module.name is not None:
                    assert module.name not in name2idx, f"duplicated name: {module.name}"
                    modules[module.name] = module
                    name2idx[module.name] = idx
                else:
                    modules[str(idx)] = module
            else:
                assert nndef[-1] is None, f"syntax error: {nndef}"
            nodeseq.append(nndef)
        self.ordered_nodes = nodeseq
        self.module_dict = nn.ModuleDict(modules)
        self.name2idx = name2idx

    def __str__(self):
        graph_str = []
        for idx, node in enumerate(self.ordered_nodes):
            node_str = []
            node_str.append("(")
            for token in node[:-1]:
                node_str.append(str(token) + ", ")
            try:
                node_str.append(node[-1].__pstr__)
            except AttributeError:
                if node[-1] is not None:
                    node_str.append(str(node[-1]))
            node_str.append(f"),  # {idx}")
            node_str = "".join(node_str)
            graph_str.append(node_str)
        graph_str = "[\n{}\n]".format("\n".join(graph_str))
        return graph_str
    
    def forward(self, inputs):
        # fill inputs into cache
        cache = []
        if not isinstance(inputs, collections.Sequence):
            inputs = [inputs]
        # forward
        for idx, node in enumerate(self.ordered_nodes):
            module = node[-1]
            if module is None:
                cache.append(inputs[idx])
            else:
                cache.append(module(cache))
        return cache

def stringfy_partial(p: partial):
    arguments = []
    for arg in p.args:
        arguments.append(str(arg))
    for key, val in p.keywords.items():
        arguments.append(f"{key}={val}")
    arguments = ", ".join(arguments)
    return f"{p.func.__name__}({arguments})"

def build(nndefs):
    return Network(nndefs)