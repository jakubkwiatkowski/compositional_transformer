"""
This module contains functions for creating transformer models that operate on grid-like inputs.

Functions:

- grid_transformer: returns a callable transformer model constructor that can operate on grid-like inputs.
"""
from typing import Optional, Tuple, Callable, Union

import core_tools.ops as K
import tensorflow as tf
from tensorflow.keras.layers import Dense

from core_tools.core import rgb, get_extractor, log_shape, Extractor
from grid_transformer.constants import TILE

from grid_transformer.utils import get_map_fn


# from tensorflow.python.keras.layers import LayerNormalization, Lambda, Conv2D, Layer,Embedding


def Tokenizer(
        batch_no: Optional[int] = None,
        col: int = 3,
        row: int = 3,
        channel: Optional[str] = None,
        extractor_input: Union[int, Tuple[int, int, int]] = 224,
        extractor: Union[str, Callable] = "ef",
        output_size: int = 10,
        pooling: Optional[Union[str, int, Callable]] = "flat2d",
        map_fn: Union[str, Callable] = "batch",
        return_extractor_input: bool = False,
        show_shape: Union[bool, Callable] = False,
        **kwargs
) -> Callable:
    """
        Applies a grid transformation to an image.

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
        last : Union[str, Callable]
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

    batch_no = batch_no
    extractor = extractor
    output_size = output_size
    pooling = pooling
    channel = channel
    map_fn = get_map_fn(map_fn)

    # def input_show(*args, **kwargs):
    #     if hasattr(args[0][0], "numpy"):
    #         ims(args[0][0].transpose((2, 1, 0)))

    # @log_shape(show_shape, "Tokenizer", in_fn=input_show)
    @log_shape(show_shape, "Tokenizer")
    def apply(x: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        nonlocal batch_no
        nonlocal extractor
        nonlocal pooling
        nonlocal channel

        shape = tf.shape(x)
        if channel == TILE:
            x = tf.transpose(x[..., None], [0, 3, 1, 2, 4])
            x = rgb((1, 1, 1, 1, 3))(x)

        else:
            x = tf.reshape(x, tf.concat([shape[:-1], [int(x.shape[-1] / 3), 3]], axis=0))
            x = tf.transpose(x, [0, 3, 1, 2, 4])

        if batch_no is None:
            batch_no = int(x.shape[1] / (row * col))

        if row > 1 or col > 1:
            x = K.create_image_grid2(x, row=row, col=col)
            x = x[:, :, :extractor_input[0], :extractor_input[1]]

        if batch_no > 1:
            extractor = map_fn(get_extractor(data=extractor_input, batch=False, model=extractor))
        else:
            x = x[:, 0]

        if callable(extractor) and pooling is None:
            y = log_shape(show_shape, "Extractor")(extractor)(x)
            if y.shape[-1] != output_size:
                y = log_shape(show_shape, "IT: Extractor pooling")(Dense(output_size))(y)

        elif extractor:
            y = Extractor(
                data=tuple(x.shape),
                model=extractor,
                projection=output_size,
                pooling=pooling,
                show_shape=show_shape,
            )(x)
        else:
            y = x

        if return_extractor_input:
            return y, x

        return y

    return apply
