"""
This module contains the `simple_transformer` function for creating a transformer model with customizable embedding options.

Functions:

- simple_transformer: Returns a transformer model constructor with the provided embedding options.
"""
from functools import partial
from typing import Union, Callable

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda, Embedding, Flatten

from core_tools.core import filter_init, log_shape, IndexReshape, il
from core_tools.flatten2d import Flatten2D
from grid_transformer.transformer import Transformer


def SimpleTransformer(
        *args,
        embed_in: int,
        size: int = 256,
        pre: Union[str, Callable] = None,
        embed: Union[str, Callable] = None,
        embed_out: Union[int, str] = 'auto',
        flatten: Union[str, Callable, bool] = False,
        show_shape: Union[bool, Callable] = False,
        **kwargs
) -> Callable:
    """
    This function returns a transformer model with the provided embedding options.

    Args:
        *args: positional arguments that will be passed to the transformer model
        embed_in (int): the input dimension of the embedding layer
        size (int): the size of the transformer model
        pre (Union[str, Callable]): pre-processing function to be applied to the input. Default is None
        embed (Union[str, Callable]): embedding function to be applied to the input. Default is None
        embed_out (Union[int, str]): the output dimension of the embedding layer. Default is 'auto'
        flatten (Union[str, Callable, bool]): flatten function to be applied to the input. Default is False
        **kwargs: keyword arguments that will be passed to the transformer model

    Returns:
        A callable transformer model functional template
    """
    if embed_out == "auto":
        embed_out = size
    transformer = filter_init(Transformer, *args, size=size, show_shape=show_shape, **kwargs)

    @log_shape(show_shape, "SimpleTransformer")
    def apply(x):
        nonlocal embed
        nonlocal flatten

        if embed == "one_hot":
            embed = Lambda(partial(tf.one_hot, depth=embed_in))
        elif embed == "linear":
            embed = Dense(embed_out)
        else:
            embed = Embedding(input_dim=embed_in, output_dim=embed_out)
        x = embed(x)
        if flatten:
            if flatten in ['2d', "flat2d"]:
                flatten = Flatten2D()
            elif il(flatten):
                flatten = IndexReshape(flatten)
            elif callable(flatten):
                flatten = flatten
            else:
                flatten = Flatten()
            x = flatten(x)
        if pre:
            x = pre(x)

        # kwargs = {
        #     **kwargs,
        #     "extractor": Sequential(embed) if len(embed) > 1 else embed[0],
        #     "extractor_pooling": None
        #
        # }
        return transformer(x)

    return apply
