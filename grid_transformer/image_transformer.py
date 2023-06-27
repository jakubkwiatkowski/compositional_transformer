"""
This module contains functions for creating transformer models that operate on grid-like inputs.

Functions:

- grid_transformer: returns a callable transformer model constructor that can operate on grid-like inputs.
"""
from typing import Optional, Tuple, Callable, Union

import tensorflow as tf

from core_tools.core import build_model, log_shape, flatten_return
from grid_transformer.preprocessor import Preprocessor
from grid_transformer.transformer import Transformer


# from tensorflow.python.keras.layers import LayerNormalization, Lambda, Conv2D, Layer,Embedding


def ImageTransformer(
        *args,
        preprocessor=None,
        transformer=None,
        pos_emd="cat",
        size=256,
        pooling=None,
        no=8,
        out_layers=(1000, 1000),
        output_size=10,
        last_norm=True,
        out_pre: Union[None, str, Tuple[str, ...]] = None,
        out_post: Union[None, str, Tuple[str, ...]] = None,
        num_heads=8,
        ff_mul=4,
        ff_size=None,
        dropout=0.1,
        ff_act="gelu",
        extractor_pooling: Optional[Union[str, int, Callable]] = "flat2d",
        save_shape=True,
        show_shape: Union[bool, Callable] = False,
        return_attention_scores=False,
        return_extractor_input: bool = False,
        **kwargs
) -> Callable:
    """
    Applies a grid transformation to an image.

    Parameters
    ----------
    extractor_shape : Union[int, Tuple[int, int, int]]
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

    preprocessor = build_model(
        preprocessor,
        *args,
        output_size=size,
        return_extractor_input=return_extractor_input,
        show_shape=show_shape,
        pooling=extractor_pooling,
        default_model=Preprocessor,
        **kwargs)

    if transformer is None:
        transformer = Transformer(
            pos_emd=pos_emd,
            size=size,
            pooling=pooling,
            no=no,
            out_layers=out_layers,
            output_size=output_size,
            last_norm=last_norm,
            out_pre=out_pre,
            out_post=out_post,
            num_heads=num_heads,
            ff_mul=ff_mul,
            ff_size=ff_size,
            dropout=dropout,
            ff_act=ff_act,
            save_shape=save_shape,
            show_shape=show_shape,
            return_attention_scores=return_attention_scores,
        )

    @log_shape(show_shape, "ImageTransformer")
    def apply(x: tf.Tensor, mask_: tf.Tensor = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        nonlocal pooling
        nonlocal transformer
        nonlocal preprocessor
        if return_extractor_input:
            x, extractor_input = preprocessor(x, mask_)
            return flatten_return(transformer(x), extractor_input)

        x = preprocessor(x, mask_)
        return transformer(x)

    return apply
