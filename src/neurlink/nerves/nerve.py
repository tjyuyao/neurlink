from __future__ import annotations
from copy import copy

import itertools
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, List, Tuple, Union
from collections.abc import Sequence

import torch
from typing_extensions import Self

from neurlink.nerves.utils import is_sequence_of, isint

from .common_types import NeurlinkAssertionError, size_any_t


class NotEnoughInputError(Exception):
    ...


_NEURLINK_META_KEY_ = "_neurlink_meta_"
NEURLINK_META_NOT_FOUND = "NEURLINK_META_NOT_FOUND"


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


class Shape:
    def __init__(
        self, size_tuple: int | Tuple[int, ...] | torch.Size, repeat_times=1
    ) -> None:
        super().__init__()
        if isint(size_tuple):
            size_tuple = tuple(itertools.repeat(int(size_tuple), repeat_times))
        self.size_tuple = size_tuple

    def __getitem__(self, index) -> Shape:
        if isinstance(index, int):
            return self.size_tuple[index]
        elif isinstance(index, slice):
            return Shape(self.size_tuple[index])
        elif isinstance(index, Sequence):
            return Shape(tuple(self.size_tuple[i] for i in index))
        else:
            raise TypeError(f"index={index}, type(index)={type(index)}")

    def numel(self) -> int:
        if isinstance(self.size_tuple, torch.Size):
            return self.size_tuple.numel()
        else:
            return prod(self.size_tuple)

    def __len__(self):
        return len(self.size_tuple)

    def __repr__(self) -> str:
        return f"Shape({self.size_tuple})"

    def __eq__(self, __o: object) -> bool:
        try:
            if len(self) != len(__o):
                return False
            for s, o in zip(self, __o):
                if s != o:
                    return False
            else:
                return True
        except:
            return False


class ShapeSpec:
    def __init__(self, expr) -> None:
        if isinstance(expr, ShapeSpec):
            other = expr
            self.expr = copy(other.expr)
            self.relative = copy(other.relative)
            self.absolute = copy(other.absolute)
        elif isinstance(expr, str):
            self.expr = expr
            self.relative: size_any_t = None
            self.absolute: Shape = Shape(eval(expr))
        elif isinstance(expr, int) or is_sequence_of(expr, int):
            self.expr = expr
            self.relative: size_any_t = expr
            self.absolute: Shape = None
        else:
            raise TypeError(f"ShapeSpec({repr(expr)})")

    def __repr__(self) -> str:
        return f"ShapeSpec({repr(self.expr)})"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, ShapeSpec):
            return self.expr == __o.expr
        else:
            return self.expr == __o

    def get_absolute(self, base_shape: size_any_t) -> Shape:
        if self.relative:
            down_scales = Shape(self.relative, repeat_times=len(base_shape))
            abs_shape = []
            for base_size, down_scale in zip(base_shape, down_scales):
                abs_size = self.size(base_size, down_scale)
                abs_shape.append(abs_size)
            return Shape(abs_shape)
        else:
            return self.absolute

    @staticmethod
    def size(base_size, down_scale) -> int:
        return base_size // down_scale + base_size % down_scale


@dataclass
class DimSpec:
    channels: int
    shape: ShapeSpec

    def __getitem__(self, index):
        return [self.channels, self.shape][index]

    def __len__(self):
        return 2

    def __eq__(self, __o: object) -> bool:
        try:
            return self[0] == __o[0] and self[1] == ShapeSpec(__o[1])
        except:
            return False


@dataclass
class NerveSpec:
    """Internal representation of nndef syntax `(dims=[(channels, shape=(size, ...)), ...], nerve)`

    Example:

    The following specification of "one layer of neural networks" (coined Nerve) in neurlink
    says a 2d convolution layer with kernel (3, 3) will ensure an output tensor of channels 64
    and shape resolution of (128, 256). The specification line itself is denoted as a `nndef`
    line. Note that the input information is specified in previous lines of nndefs and not
    demonstrated here.

    ```python
    nndef = ((64, "(128, 256)"), nv.Conv2d(3))
    dims, nerve = nndef[:-1], nndef[-1]
    for channels, shape in dims:
        for size in eval(shape):
            ...
    ```
    """

    dims: List[DimSpec]
    nerve: Nerve


class _NerveRegistry:
    def __init__(self, name) -> None:
        self.name = name
        self.registry = {}

    def register(self, cls, container_nerve: Nerve, selector, target_dims):
        """produces a pickable dynamic subclass of `cls` born with meta-info available during __init__."""

        # input_selector
        if selector is None:
            input_selector, tag = None, None
        elif isinstance(selector, tuple) and len(selector) == 2:
            input_selector, tag = selector
        elif isinstance(selector, str) and selector not in container_nerve:
            input_selector, tag = None, selector
        elif isinstance(selector, (int, str, slice, Sequence)):
            input_selector, tag = selector, None
        else:
            raise NeurlinkAssertionError(f"TypeError(selector={selector})")

        if isinstance(input_selector, (int, str)):
            input_selector = [input_selector]

        ntypename = f"{cls.__name__}{len(self.registry)}"

        if cls is None or cls is Input:
            cls = Input
            input_links = None
        else:
            input_links = container_nerve[input_selector]

        def dynamic_subclass_init(self, *args, _Nerve__finalized, **kwds):
            assert _Nerve__finalized is True
            cls.__init__(self, *args, **kwds)

        ntype = type(
            f"{self.name}.{ntypename}",
            (cls,),
            {
                "__init__": dynamic_subclass_init,
                _NEURLINK_META_KEY_: dict(
                    input_links=input_links,
                    target_dims=target_dims,
                    input_selector=input_selector,
                    container_nerve=container_nerve,
                    tag=tag,
                ),
            },
        )

        self.registry[ntypename] = ntype
        return ntype

    def __getattr__(self, ntypename):  # for pickling
        return self.registry[ntypename]


# `_NB` is the single unique instance of `_NerveRegistry` class.
_NB = _NerveRegistry("_NB")


class Nerve(torch.nn.Module):
    """Base class to support input selection syntax (Typename[slice, name]).

    Attributes:
        input_links (List[`NerveSpec`]): [i].
        target_dims (List[`DimSpec`]): [i].
        base_shape  (`Shape`): [f].
        output_shapes  (List[Tuple[int, ...]]): [f].
        nerves (List[`NerveSpec`]): [i].
    """

    def __new__(
        cls: type[Self], *args, _Nerve__finalized=False, **kwds
    ) -> Self:  # default option when selector omitted
        if _Nerve__finalized:
            return super().__new__(cls)
        else:
            return cls[-1](*args, **kwds)

    def __class_getitem__(cls, selector):  # input_selector call
        def parameter_keeper(*args, **kwds):  # constructor call

            # finalize call
            def nerve_builder(container_nerve: Nerve, target_dims):
                nerve_dynamic_class = _NB.register(
                    cls, container_nerve, selector, target_dims
                )
                # dynamic class is also a subclass of Nerve,
                # `_Nerve__finalized=True` will ensure a real instantiation.
                nerve_object = nerve_dynamic_class(
                    *args, _Nerve__finalized=True, **kwds
                )
                return nerve_object

            return nerve_builder

        return parameter_keeper

    @property
    def input_links(self) -> List[NerveSpec]:
        return getmeta(self, "input_links")

    @property
    def target_dims(self) -> List[DimSpec]:
        return getmeta(self, "target_dims")

    @property
    def output_shapes(self) -> List[Shape]:
        return [t.shape.get_absolute(self.base_shape) for t in self.target_dims]

    @property
    def base_shape(self) -> Shape:
        return self._ensure_inputs.base_input_nerve.shape

    def add(self, nndef, tag=None) -> int:

        if isinstance(nndef, list):
            return [self.add(n) for n in nndef]

        if isinstance(nndef, NerveSpec):
            target_dims = nndef.dims
            nerve_builder = nndef.nerve
        elif isinstance(nndef, tuple):
            target_dims, nerve_builder = nndef[:-1], nndef[-1]
            target_dims = [DimSpec(c, ShapeSpec(s)) for c, s in target_dims]
        else:
            raise TypeError(f"nndef={repr(nndef)}")

        # retrieve actual modules.
        if getattr(nerve_builder, "__name__", None) == "nerve_builder":
            """
            pass current network and target_dims of next module to add to nerve_builder.
            The nerve_builder will register a new dynamic class in _NerveRegistry so that
            meta infomation defined by the nndef syntax is populated before user's __init__
            function. Then the nerve_builder initializes a new module and return it back.

            Following is the pseudo_code for nerve_builder:

            ```python
            def nerve_builder(ctx, container_nerve, target_dims):
                input_links = container_nerve[ctx.input_selector]
                dynamic_class = type(
                    ctx.class_name,
                    (ctx.target_class,),
                    {
                        "input_links": input_links,
                        "target_dims": target_dims,
                        "input_selector": ctx.input_selector,
                        "container_nerve": container_nerve,
                        "tag": ctx.tag,
                    },
                )
                return dynamic_class
            ```
            """
            nerve_object = nerve_builder(self, target_dims)
            tag = getmeta(nerve_object, "tag") or tag
        elif isinstance(nerve_builder, (torch.nn.Module, Callable)):
            # support to `nn.Module` and generic callable is an experimental feature and currently
            # not recommended for production code. But this is an option for maximum compatibility.
            # and flexibility. An `nn.Module` will receive the last element in previous sequence as
            # input for `forward` function, while a generic callable will receive the whole cache.
            nerve_object = nerve_builder
        else:
            raise TypeError(nerve_builder)

        # register each module for later lookup.
        idx = len(self.nerves)
        tag = tag if tag is not None else str(idx)
        self.nerves.append(NerveSpec(target_dims, nerve_object))
        self._tag2idx[tag] = idx
        self._module_dict[tag] = nerve_object
        self._ensure_inputs(nerve_object)
        return idx

    class _EnsureInputBaseShape:
        """Ensure an Input as base_shape reference; as well as propagating input_links to sub-nerve.

        If the target nerve is a top-level nerve built from a nndef syntax, its element
        nerves added by `Nerve.add(...)` will call this functor to find the first Input
        specified `base=True` by user or set `base=True` for the first Input nerve if
        user has not specified any.

        If the target nerve is a sub-level nerve initialized with meta "container_nerve"
        available, due to the global property of base_shape, this class will use the same
        Input reference from the container_nerve as the reference of the target sub-nerve.

        Meantime, for the sub-nerve case, the container_nerve's input_links will be propagated
        on at construction. Such that, the sub-nerve will have proper Input nodes in its
        node_sequence

        Attributes:
            base_input_nerve (Input): the base_shape provider Input.
            nerves (List[`NerveSpec`]): ...
        """

        def __init__(self, nerve: Nerve) -> None:
            if isinstance(nerve, Input):  # Input prohibited base_shape attribute.
                self.base_input_nerve: Input = None
                self.done = True
            elif not hasmeta(nerve, "container_nerve"):  # top-level
                self.base_input_nerve: Input = None
                self.done = False
            else:  # sub-level
                # reuse global base_shape_input_provider
                container_nerve: Nerve = getmeta(nerve, "container_nerve")
                self.base_input_nerve: Input = (
                    container_nerve._ensure_inputs.base_input_nerve
                )
                assert self.base_input_nerve is not None
                self.done = True

        def __call__(self, sub_nerve):
            if self.done:
                return
            if isinstance(sub_nerve, Input):
                if sub_nerve.base_shape_flag:
                    self.base_input_nerve = sub_nerve
                    self.done = True
                    return
                if self.base_input_nerve is None:
                    self.base_input_nerve = sub_nerve
            else:  # first non-input nerve
                if self.base_input_nerve is None:
                    raise ValueError(
                        "No Input nerves found! You should add them at the very first."
                    )
                else:
                    self.base_input_nerve.base_shape_flag = True
                    self.done = True
                    return

    def __init__(self, *args, _Nerve__finalized=False, **kwds) -> None:
        super().__init__()
        self.nerves: List[NerveSpec] = []
        self._tag2idx: Dict[str, int] = {}
        self._module_dict = torch.nn.ModuleDict()
        self._ensure_inputs = Nerve._EnsureInputBaseShape(self)

        # input propagation through hierarchy
        if hasmeta(self, "container_nerve"):  # sub-level
            input_links = self.input_links
            if not isinstance(input_links, list):
                input_links = [input_links]
            for input_link in input_links:
                target_dims = input_link.dims
                self.add((*target_dims, Input()))

    def _get_sequence_item(self, sequence, index):
        """impl. of `sequence[index]` but allows index to be/contain tagname."""
        if index is None:
            return []
        elif isinstance(index, int):
            return sequence[index]
        elif isinstance(index, str):
            return sequence[self._tag2idx[index]]
        elif isinstance(index, slice):
            args = dict(start=None, stop=None, step=None)
            for attrname in ("start", "stop", "step"):
                attrvalue = getattr(index, attrname)
                if isinstance(attrvalue, str):
                    args[attrname] = self._tag2idx[attrvalue]
                else:
                    args[attrname] = attrvalue
            new_index = slice(args["start"], args["stop"], args["step"])
            return sequence[new_index]
        elif isinstance(index, list):
            return [self._get_sequence_item(sequence, i) for i in index]
        elif isinstance(index, set):
            return {i: self._get_sequence_item(sequence, i) for i in index}
        else:
            raise TypeError(f"invalid input range {index} of type {type(index)}")

    def __getitem__(self, index):
        return self._get_sequence_item(self.nerves, index)

    def __contains__(self, index):
        if isinstance(index, (str, int)):
            try:
                self._get_sequence_item(self.nerves, index)
                return True
            except KeyError:
                return False
        return False

    def __len__(self):
        return len(self.nerves)

    def forward(self, inputs, output_list=False):
        # fill inputs into cache
        cache = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        # forward
        for idx, node in enumerate(self.nerves):
            module = node.nerve
            if isinstance(module, Input):
                try:
                    output = inputs[idx]
                except IndexError:
                    raise NotEnoughInputError(
                        f"while extracting {idx}-th (zero-based) Input."
                    )
                # populate Input.shape at runtime
                if module.base_shape_flag:
                    if isinstance(output, torch.Tensor):
                        module.shape = Shape(output.shape)
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
        if output_list:
            return cache
        else:
            return cache[-1]

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.Module]:
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'. "
        if name in ["nerves", "input_links", "taget_dims"]:
            msg += "\n[Nerve] Please make sure call super().__init__() first."
        raise AttributeError(msg)


class Input(Nerve):
    """Placeholder that bypasses input sequence elements from container-nerve to sub-nerve."""

    def __init__(self, base_shape: bool = False, example: torch.Tensor = None) -> None:
        torch.nn.Module.__init__(self)
        self.base_shape_flag = base_shape
        self.example = example
        self.shape: Shape = None

    @property
    def base_shape(self) -> None:
        ...

    def __str__(self) -> str:
        tag = getmeta(self, "tag")
        tag = "" if tag is None else f", {tag}"
        return f"Input[{getmeta(self, 'input_selector')}{tag}]()"


def build(nndefs: list):
    def expand_sublist(nndef):
        if isinstance(nndef, list):
            for i in nndef:
                yield from expand_sublist(i)
        else:
            yield nndef

    assert isinstance(nndefs, list)
    network = Nerve(_Nerve__finalized=True)
    for nndef in expand_sublist(nndefs):
        network.add(nndef)
    return network
