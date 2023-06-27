"""
This module contains utility functions for creating and manipulating masks for grid-like inputs.

Functions:

- init_weights: Initialize weights for a tensor with a shape of `shape` and data type of `dtype`.
- take_left: Select the leftmost column of the tensor x.
- take_by_index: Select a column of the tensor x by index
- mix: Mix the leftmost and the middle column of the tensor x
- empty_last: Returns a tensor of zeros with the same shape as the last column of x
- random_mask: Generates a random mask with the same shape as the input tensor
- take_by_index: Take a slice of an array by index
- mix: Mix 2 slices of an array by index

Example usage:

    # Initialize weights for a tensor of shape (3,3,3)
    weights = init_weights(shape=(3,3,3))

    # Select the leftmost column of a tensor of shape (batch_size, height, width, channels)
    left_column = take_left(x)

    # Select a column of the tensor x by index
    column = take_by_index(x, i=4)

    # Mix the leftmost and the middle column of the tensor x
    mixed = mix(x)

    # Generates a random mask with the same shape as the input tensor
    mask = random_mask(inputs, last_index=5)

    # Take a slice of an array by index
    sliced = take_by_index(x, i=4)

    # Mix 2 slices of an array by index
    mixed = mix(x)
"""

from typing import Tuple, Callable, Optional

import tensorflow as tf
from core_tools import ops as K


def init_weights(shape: Tuple[int], dtype: tf.DType = tf.float32) -> tf.Tensor:
    """
    Initialize weights for a tensor with a shape of `shape` and data type of `dtype`
    :param shape: Tuple of integers representing the shape of the tensor.
    :param dtype: Data type of the tensor. Default is tf.float32
    :return: Tensor with the initialized weights
    """
    return tf.cast(K.init.image(shape=shape, pre=True), dtype=dtype)


def take_left(x: tf.Tensor) -> tf.Tensor:
    """
    Select the leftmost column of the tensor x
    :param x: Tensor of shape (batch_size, height, width, channels)
    :return: Tensor of shape (batch_size, height, 1, channels)
    """
    return x[..., 7:8]


def take_by_index(x: tf.Tensor, i: int = 8) -> tf.Tensor:
    """
    Take a slice of an array by index.

    Parameters:
        x (tf.Tensor): The input array.
        i (int, optional): The index of the slice. Default is 8.

    Returns:
        tf.Tensor: The sliced array.
    """
    return x[..., i:i + 1]


def mix(x: tf.Tensor) -> tf.Tensor:
    """
    Mix 2 slices of an array by index.

    Parameters:
        x (tf.Tensor): The input array.

    Returns:
        tf.Tensor: The mixed array.
    """
    return (x[..., 7:8] + x[..., 5:6]) / 2


def empty_last(x: tf.Tensor) -> tf.Tensor:
    """
    Return a tensor of zeros with the shape of the last slice of input tensor.

    Parameters:
        x (tf.Tensor): The input tensor.

    Returns:
        tf.Tensor: The tensor of zeros.
    """
    return tf.zeros_like(x[..., 7:8])


def random_mask(inputs: tf.Tensor, last_index: int) -> tf.Tensor:
    """
    Generate a random mask of the same shape as inputs.

    Parameters:
        inputs (tf.Tensor): The input tensor.
        last_index (int): The last index to use for generating the mask.

    Returns:
        tf.Tensor: A boolean tensor of the same shape as inputs, with random values of True or False.
    """
    shape = tf.shape(inputs)
    indexes = tf.random.uniform(shape=shape[0:1], maxval=last_index, dtype=tf.int32)
    mask = tf.cast(tf.one_hot(indexes, last_index), dtype='bool')
    return mask


# todo docs
def no_mask(inputs: tf.Tensor, last_index: int) -> tf.Tensor:
    """
    Generate a random mask of the same shape as inputs.

    Parameters:
        inputs (tf.Tensor): The input tensor.
        last_index (int): The last index to use for generating the mask.

    Returns:
        tf.Tensor: A boolean tensor of the same shape as inputs, with random values of True or False.
    """
    shape = tf.shape(inputs)
    # Unnecessary cast to have same number of ops as other function mask. This allow to load weights.
    mask = tf.cast(tf.zeros(tf.concat([shape[0:1], tf.convert_to_tensor([last_index])], axis=-1), dtype='bool'), dtype='bool')
    return mask


def constant_mask(inputs: tf.Tensor, value: int = 8, last_index: Optional[int] = None) -> tf.Tensor:
    """
    Generate a constant mask of the same shape as inputs.
    The mask will have the same value in all positions.

    Parameters:
        inputs (tf.Tensor): The input tensor.
        value (int, optional): The value to fill the mask with. Default is 8.
        last_index (int, optional): The last index to use for generating the mask. If not provided, it will be set to value + 1.

    Returns:
        tf.Tensor: A boolean tensor of the same shape as inputs, with constant value of True or False.
    """
    shape = tf.shape(inputs)
    indexes = tf.fill(dims=shape[0:1], value=value)
    mask = tf.cast(tf.one_hot(indexes, last_index if last_index else value + 1), dtype='bool')
    return mask


class ImageMask(tf.keras.Model):
    def __init__(self, mask_value: Callable[[tf.Tensor], tf.Tensor], last_index: Optional[int] = None):
        """
        Class that applies a mask to an image.

        Parameters:
            mask_value (Callable[[tf.Tensor], tf.Tensor]): A callable that takes an image tensor and returns the last slice.
            last_index (int, optional): The last index of the image. If not provided, it will be set to the last index of the image.
        """
        super().__init__()
        self.mask_value = mask_value
        self.last_index = last_index

    # noinspection PyMethodOverriding
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Applies the mask to the image.

        Parameters:
            inputs (Tuple[tf.Tensor, tf.Tensor]): A tuple containing the image tensor and the mask tensor.

        Returns:
            tf.Tensor: The masked image.
        """
        inputs, mask = inputs
        mask = mask[:, None, None]
        inputs = inputs[..., :self.last_index]
        return tf.cast((1 - mask) * inputs + mask * tf.tile(self.mask_value(inputs)[None], (1, 1, 1, tf.shape(inputs)[-1])),
                       dtype='float32')
