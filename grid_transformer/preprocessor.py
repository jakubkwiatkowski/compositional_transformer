"""
This module contains functions for creating transformer models that operate on grid-like inputs.

Functions:

- grid_transformer: returns a callable transformer model constructor that can operate on grid-like inputs.
"""
import math
from typing import Optional, Tuple, Callable, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda

from core_tools.core import InitialWeight, log_shape, build_model
from grid_transformer.constants import LEFT, MIX, EMPTY, LEARNABLE
from grid_transformer.mask import ImageMask, take_left, mix, empty_last, init_weights
from grid_transformer.tokenizer import Tokenizer


# from tensorflow.python.keras.layers import LayerNormalization, Lambda, Conv2D, Layer,Embedding


def Preprocessor(
        batch_no: Optional[int] = None,
        col: int = 3,
        row: int = 3,
        channel: Optional[str] = None,
        extractor_input: Union[int, Tuple[int, int, int]] = 224,
        *args,
        tokenizer: Optional[tf.keras.Model] = None,
        mask_value: Union[str, Callable] = "start",
        last_index: Optional[int] = None,
        show_shape: Union[bool, Callable] = False,
        **kwargs
) -> Callable:
    """
    Class that wrap tokenizer with additional functionality, like masking or resizing input.

    Parameters
    ----------
    extractor_input : Union[int, Tuple[int, int, int]]
        Shape of the image extractor.
    batch_no : Optional[int]
        Number of batches to be used.
    col : int
        Number of columns in the grid.
    row : int
        Number of rows in the grid.
    no : int
        Number of attention heads.
    extractor : Union[str, Callable]
        Extractor to be used for extracting features from the image.
    output_size : int
        Size of the output of the transformer model.
    pos_emd : str
        Type of positional encoding to be used.
    mask_value : Union[str, Callable]
        Last layer of the transformer model.
    pooling : Optional[Union[str, int, Callable]]
        Pooling function to be used.
    model : Optional[tf.keras.Model]
        Transformer model to be used.
    map_fn : Union[str, Callable]
        Function to map the extracted features to the transformer model.
    channel : Optional[str]
        How channels are handled.
    last_index : Optional[int]
        Index of the last layer.
    return_extractor_input : bool
        If True, returns the input of the extractor.
    return_attention_scores : bool
        If True, returns the attention scores.
    **kwargs :
        Additional keyword arguments to be passed to the transformer model.

    Returns
    -------
    Callable
        Grid Transformer constructor.
    """
    extractor_input = (extractor_input, extractor_input, 3) if isinstance(extractor_input,
                                                                          int) else extractor_input

    image_shape = (math.ceil(extractor_input[0] / col), math.ceil(extractor_input[0] / row))

    if mask_value == LEFT:
        mask_value = take_left
    elif mask_value == MIX:
        mask_value = mix
    elif mask_value == EMPTY:
        mask_value = empty_last
    elif mask_value == LEARNABLE:
        mask_value = Sequential([Lambda(empty_last), InitialWeight(initializer=init_weights)])

    tokenizer = build_model(
        tokenizer,
        batch_no=batch_no,
        col=col,
        row=row,
        channel=channel,
        extractor_input=extractor_input,
        show_shape=show_shape,
        *args,
        default_model=Tokenizer,
        **kwargs
    )

    last_index = last_index

    @log_shape(show_shape, "Preprocessor")
    def apply(x: tf.Tensor, mask_: tf.Tensor) -> tf.Tensor:
        x = tf.image.resize(tf.transpose(x, (0, 2, 3, 1)), image_shape)
        # x = tf.transpose(x, (0, 2, 3, 1))
        x = ImageMask(mask_value=mask_value, last_index=last_index)((x, mask_))
        return tokenizer(x)

    return apply
