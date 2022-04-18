from typing import TypeVar, Union, Tuple, Optional
from torch import Tensor

# Create some useful type aliases

# Template for arguments which can be supplied as a tuple, or which can be a scalar which PyTorch will internally
# broadcast to a tuple.
# Comes in several variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d operations.
T = TypeVar("T")
scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
scalar_or_tuple_1_t = Union[T, Tuple[T]]
scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
size_any_t = scalar_or_tuple_any_t[int]
size_1_t = scalar_or_tuple_1_t[int]
size_2_t = scalar_or_tuple_2_t[int]
size_3_t = scalar_or_tuple_3_t[int]
size_4_t = scalar_or_tuple_4_t[int]
size_5_t = scalar_or_tuple_5_t[int]
size_6_t = scalar_or_tuple_6_t[int]

# For arguments which represent optional size parameters (eg, adaptive pool parameters)
size_any_opt_t = scalar_or_tuple_any_t[Optional[int]]
size_2_opt_t = scalar_or_tuple_2_t[Optional[int]]
size_3_opt_t = scalar_or_tuple_3_t[Optional[int]]

# For arguments that represent a ratio to adjust each dimension of an input with (eg, upsampling parameters)
ratio_2_t = scalar_or_tuple_2_t[float]
ratio_3_t = scalar_or_tuple_3_t[float]
ratio_any_t = scalar_or_tuple_any_t[float]

tensor_list_t = scalar_or_tuple_any_t[Tensor]

# For the return value of max pooling operations that may or may not return indices.
# With the proposed 'Literal' feature to Python typing, it might be possible to
# eventually eliminate this.
maybe_indices_t = scalar_or_tuple_2_t[Tensor]

# some common exceptions


class NeurlinkAssertionError(Exception):
    def __init__(self, *args: object) -> None:
        args = (
            "Please report a bug at https://github.com/tjyuyao/neurlink/issues :",
            *args,
        )
        super().__init__(*args)


class NNDefParserError(Exception): ...