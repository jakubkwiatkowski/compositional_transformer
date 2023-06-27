import functools

import numpy as np
from ml_utils import il, format_
from models_utils import BatchModel, Pooling, log_shape

from grid_transformer.constants import VEC
from grid_transformer.layers import Vec, Detokenizer


# Keras Rehsappe _fix_unknown_dimension
def inference_shape(input_shape, output_shape):
    """Find and replace a missing dimension in an output shape.

    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

    Args:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most a single -1
        which indicates a dimension that should be derived from the input
        shape.

    Returns:
      The new output shape with a -1 replaced with its computed value.

    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
    output_shape = list(output_shape)
    msg = (
        "total size of new array must be unchanged, "
        "input_shape = {}, output_shape = {}".format(
            input_shape, output_shape
        )
    )

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
        if dim < 0:
            if unknown is None:
                unknown = index
            else:
                raise ValueError(
                    "There must be at most one unknown dimension in "
                    f"output_shape. Received: output_shape={output_shape}."
                )
        else:
            known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
        if known == 0 or original % known != 0:
            raise ValueError(msg)
        output_shape[unknown] = original // known
    elif original != known:
        raise ValueError(msg)
    return output_shape


def get_map_fn(map_fn):
    if map_fn == VEC:
        return Vec
    elif callable(map_fn):
        return map_fn
    else:
        return BatchModel


def MultiPooling(pooling, *args, show_shape=False, **kwargs):
    if isinstance(pooling, int):
        pooling = Detokenizer(pooling, *args, **kwargs)
        if show_shape:
            return log_shape(show_shape, "Detokenizer")(pooling)
        return pooling
    return Pooling(pooling, *args, show_shape=show_shape, **kwargs)
