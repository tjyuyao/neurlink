import collections
from functools import partial, update_wrapper
from itertools import repeat
from typing import Callable, List, Dict, Any
from typing_extensions import Self


def ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            "Input dimension should be at least {}".format(len(out_size) + 1)
        )
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size) :])
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

def isint(x):
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return x - int(x) == 0.0
    return False


def expect_int(x):
    if isint(x):
        return int(x)
    raise TypeError(f"Expecting a int, got a {type(x)}.")


def is_sequence_of(seq, types):
    for x in seq:
        if not isinstance(x, types):
            return False
    return True

all_type = is_sequence_of

def all_sequence_of(seq, types):
    for x in seq:
        if not is_sequence_of(x, types):
            return False
    return True

class specialize(partial):
    def __getitem__(self, key):
        return specialize(self.func[key], *self.args, **self.keywords)


class _reprable:
    """Decorates a function with a repr method."""

    def __init__(self, wrapped, custom_repr):
        self._wrapped = wrapped
        self.custom_repr = custom_repr
        update_wrapper(self, wrapped)

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def __repr__(self):
        return self.custom_repr


def reprable(custom_repr:str):
    """Decorates a function with a repr method."""
    def decorator(f):
        return _reprable(f, custom_repr)
    return decorator

def format_args_kwds(*args, **kwds):
    fmt_args = ', '.join([repr(x) for x in args])
    fmt_kwds = ', '.join([f"{k}={repr(v)}" for k, v in kwds.items()])
    return ', '.join([x for x in (fmt_args, fmt_kwds) if x])