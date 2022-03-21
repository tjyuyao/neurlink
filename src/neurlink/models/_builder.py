from ast import arguments
import collections
from copy import copy
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


class Network(nn.Module):

    def __init__(self, nndefs) -> None:
        super().__init__()
        nodeseq = []
        modules = {}
        name2idx = {}
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