#%%

from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import Self

import torch


def expand(nndef):
    if isinstance(nndef, list):
        for i in nndef:
            yield from expand(i)
    else:
        yield nndef


class Network(torch.nn.Module, Sequence):
    """A `torch.nn.Module` that contains an intra-linked sequence (a.k.a directed acyclic graph) of Nerves."""

    def __init__(self, nndef):
        super().__init__()
        self.node_sequence = []
        self.tag2idx = {}
        modules = {}
        for idx, defitem in enumerate(expand(nndef)):
            target_shapes, nerve_builder = defitem[:-1], defitem[-1]
            if nerve_builder.__name__ != "nerve_builder":
                nerve_object = nerve_builder(self, target_shapes)
                tag = nerve_object._neurlink_meta_["tag"]
            else:
                nerve_object = nerve_builder(self, target_shapes)
                tag = None

            tag = tag if tag is not None else str(idx)
            self.node_sequence.append(list(target_shapes) + [nerve_object])
            self.tag2idx[tag] = idx
            modules[tag] = nerve_object
        self.module_dict = torch.nn.ModuleDict(modules)

    def _get_sequence_item(self, sequence, index):
        """impl. of `sequence[index]` but allows index to be/contain tagname."""
        if isinstance(index, int):
            return sequence[index]
        elif isinstance(index, str):
            return sequence[self.tag2idx[index]]
        elif isinstance(index, slice):
            for attrname in ("start", "stop", "step"):
                attrvalue = getattr(index, attrname)
                if isinstance(attrvalue, str):
                    setattr(index, attrname, self.tag2idx[attrvalue])
            return sequence[index]
        elif isinstance(index, list):
            return [self._get_sequence_item(sequence, i) for i in index]
        elif isinstance(index, set):
            return {i: self._get_sequence_item(sequence, i) for i in index}
        else:
            raise TypeError(f"invalid input range {index} of type {type(index)}")

    def __getitem__(self, index):
        return self._get_sequence_item(self.node_sequence, index)

    def __len__(self):
        return len(self.node_sequence)

    def forward(self, inputs):
        # fill inputs into cache
        cache = []
        if not isinstance(inputs, Sequence):
            inputs = [inputs]
        # forward
        for idx, node in enumerate(self.node_sequence):
            module = node[-1]
            if module is None or isinstance(module, Required):
                cache.append(inputs[idx])
            elif isinstance(module, Nerve):
                cache.append(
                    module(
                        self._get_sequence_item(
                            cache, module._neurlink_meta_["input_selector"]
                        )
                    )
                )
            else:
                cache.append(module(cache[-1]))
        return cache


class _NerveRegistry:
    """Helper class to generate pickable dynamic sub-subclass of `Nerve` that has `input_links` attribute."""

    def __init__(self) -> None:
        self.registry = {}

    def register(
        self, cls, input_network: Network, selector, target_shapes
    ):  # for Nerve.__class_getitem__ calling

        if selector is None:
            input_selector, tag = -1, None
        elif isinstance(selector, tuple) and len(selector) == 2:
            input_selector, tag = selector
        elif isinstance(selector, str) and selector not in input_network:
            input_selector, tag = None, selector
        else:
            input_selector, tag = selector, None

        ntypename = f"{cls.__name__}{len(self.registry)}"

        if cls is None or cls is Required:
            cls = Required
            input_links = input_network[:]
        else:
            input_links = input_network[input_selector]

        def dynamic_subclass_init(self, *args, from_dynamic_subclass=True, **kwds):
            cls.__init__(self, *args, **kwds)

        ntype = type(
            f"_NB.{ntypename}",
            (cls,),
            {
                "__init__": dynamic_subclass_init,
                "_neurlink_meta_": dict(
                    input_links=input_links,
                    target_shapes=target_shapes,
                    input_selector=input_selector,
                    tag=tag,
                ),
            },
        )

        self.registry[ntypename] = ntype
        return ntype

    def __getattr__(self, ntypename):  # for pickling
        return self.registry[ntypename]


# `_NB` is the single unique instance of `_NerveRegistry` class.
_NB = _NerveRegistry()


class Nerve(torch.nn.Module):
    """Base class to support input selection syntax (Typename[slice, name])."""

    def __new__(cls: type[Self], *args, from_dynamic_subclass=False, **kwds) -> Self:  # default option when selector omitted
        if from_dynamic_subclass:
            return super().__new__(cls)
        else:
            return cls[-1](*args, **kwds)

    def __class_getitem__(cls, selector):
        def parameter_keeper(*args, **kwds):  # for user init call
            def nerve_builder(
                input_network: Network, target_shapes
            ):  # for neurlink.build() call
                nerve_dynamic_class = _NB.register(
                    cls, input_network, selector, target_shapes
                )
                nerve_object = nerve_dynamic_class(*args, from_dynamic_subclass=True, **kwds)
                return nerve_object

            return nerve_builder

        return parameter_keeper

    @property
    def input_links(self):
        return self._neurlink_meta_["input_links"]

    @property
    def target_shapes(self):
        return self._neurlink_meta_["target_shapes"]

class Required(Nerve):
    """Effectively do nothing."""

    pass


class Select(Nerve):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


def _nerve_spawn():

    cfg = Select[1]("p1")
    m = cfg("full_input_links")
    print(isinstance(m, Select))
    print(m.param1)
    print(m.input_links)

    # # dill.loads(dill.dumps(m))
    # import pickle
    # n = pickle.loads(pickle.dumps(m))
    # m.param1 = "pp"
    # print(n.param1)

    # nerve = Nerve()
    # nerve[:]  # all
    # nerve[-1]  # number
    # nerve["-1"]  # number
    # nerve[-2:-1]  # number range
    # nerve[-2::-1]  # number range
    # nerve["-1"]  # extract
    # nerve["n1"]  # tag
    # nerve["n1":"n2", "this"]  # tag
    # nerve["n1":"n2-1", "this"]  # offset
    # nerve["n1":"n2[1]", "this"]  # repeat
    # nerve[["n1","n2", "n3+1"], "this"]
    # TODO: test input_links 唯一性
    # TODO: test w/ or w/o class_getitem


if __name__ == "__main__":
    test_nerve_spawn()
