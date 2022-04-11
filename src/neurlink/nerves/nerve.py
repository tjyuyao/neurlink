#%%

from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Tuple
from typing_extensions import Self

import torch


_NEURLINK_META_KEY_ = "_neurlink_meta_"
NEURLINK_META_NOT_FOUND = "NEURLINK_META_NOT_FOUND"


def expand(nndef):
    if isinstance(nndef, list):
        for i in nndef:
            yield from expand(i)
    else:
        yield nndef


def setmeta(obj, key, value):
    if not hasattr(obj, _NEURLINK_META_KEY_):
        setattr(obj, _NEURLINK_META_KEY_, {})
    getattr(obj, _NEURLINK_META_KEY_)[key] = value


def getmeta(obj, key, default=NEURLINK_META_NOT_FOUND):
    try:
        return getattr(obj, _NEURLINK_META_KEY_)[key]
    except:
        return default


def hasmeta(obj, key):
    try:
        return key in getattr(obj, _NEURLINK_META_KEY_)
    except:
        return False


def setmetadefault(obj, key, default):
    if not hasmeta(obj, key):
        setmeta(obj, key, default)


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

        if cls is None or cls is Input:
            cls = Input
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
                _NEURLINK_META_KEY_: dict(
                    input_links=input_links,
                    target_shapes=target_shapes,
                    input_selector=input_selector,
                    container_network=None,
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

    def __new__(
        cls: type[Self], *args, from_dynamic_subclass=False, **kwds
    ) -> Self:  # default option when selector omitted
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
                nerve_object = nerve_dynamic_class(
                    *args, from_dynamic_subclass=True, **kwds
                )
                return nerve_object

            return nerve_builder

        return parameter_keeper

    @property
    def input_links(self):
        return getmeta(self, "input_links")

    @property
    def target_shapes(self) -> Tuple[int, ...]:
        return getmeta(self, "target_shapes")
    
    @property
    def base_shape(self) -> Tuple[int, ...]:
        return getmeta(self, "container_network").required_base


class Network(torch.nn.Module, Sequence):
    """A container that contains an intra-linked sequence (a.k.a directed acyclic graph) of Nerves."""

    def __init__(self, nndef):
        super().__init__()
        self.node_sequence = []
        self.tag2idx = {}
        modules = {}

        first_required = None
        required_base1 = False
        find_user_set1 = True

        for idx, defitem in enumerate(expand(nndef)):

            # retrieve actual modules.
            target_shapes, nerve_builder = defitem[:-1], defitem[-1]
            if nerve_builder.__name__ == "nerve_builder":
                # pass current network and target_shapes of next module to add to nerve_builder.
                # The nerve_builder will register a new dynamic class in _NerveRegistry so that
                # meta infomation defined by the nndef syntax is populated before user's __init__
                # function. Then the nerve_builder initializes a new module and return it back.
                nerve_object = nerve_builder(self, target_shapes)
                tag = getmeta(nerve_object, "tag")
                setmeta(nerve_object, "container_network", self)
            elif isinstance(nerve_builder, (torch.nn.Module, Callable)):
                # support to `nn.Module` and generic callable is an experimental feature and currently
                # not recommended for production code. But this is an option for maximum compatibility.
                # and flexibility. An `nn.Module` will receive the last element in previous sequence as
                # input for `forward` function, while a generic callable will receive the whole cache.
                nerve_object = nerve_builder
                tag = None
            else:
                raise TypeError(nerve_builder)

            # register each module for later lookup.
            tag = tag if tag is not None else str(idx)
            self.node_sequence.append(list(target_shapes) + [nerve_object])
            self.tag2idx[tag] = idx
            modules[tag] = nerve_object

            # Record first `Required` object, and find out whether user has set `base=True` before any consumption.
            if find_user_set1 and isinstance(nerve_object, Input):
                if first_required is None:
                    first_required = nerve_object
                if nerve_object.base:
                    required_base1 = True
            else:
                find_user_set1 = False
        
        # Ensure at least one `Required` specified `base=True` before any cosumption.
        if not required_base1:
            first_required.base = True

        self.module_dict = torch.nn.ModuleDict(modules)
        self.required_base = None

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
            if isinstance(module, Input):
                output = inputs[idx]
                # populate required_base for container_network
                if module.base:
                    if isinstance(output, torch.Tensor):
                        self.required_base = output.shape
                    else:
                        raise TypeError(type(output))
            elif isinstance(module, Nerve):
                cache_selector = module._neurlink_meta_["input_selector"]
                cache_selected = self._get_sequence_item(cache, cache_selector)
                output = module(cache_selected)
            elif isinstance(module, torch.nn.Module):
                output = module(cache[-1])
            elif isinstance(module, Callable):
                output = module(cache)
            else:
                raise TypeError(module)
            cache.append(output)
        return cache


class Input:
    """Placeholder that bypasses input sequence elements for a Network."""
    def __init__(self, base=False) -> None:
        super().__init__()
        self.base = base


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
