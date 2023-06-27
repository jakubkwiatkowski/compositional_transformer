"""
This module contains functions for preprocessing input data for use in a transformer model.
The functions in this module are used to create the initial grid for the model.

The `get_matrix` function takes in an input tensor and an index tensor, and returns a new tensor that is a concatenation of the first `last_index` elements of the input tensor, and the element at the index specified in the index tensor. The `last_index` parameter defaults to 8.

The `get_images` function takes in a tuple containing the input tensor and the indices tensor, and returns the output tensor obtained by calling the `get_matrix` function with these inputs. The `last_index` parameter defaults to 8.

The `get_images_with_index` function takes in an input tensor, and an index and last_index and returns a tensor where the last element of the input tensor is replaced with the element at the specified index.

The `random_last` function takes in an input tensor and a last_index and returns a tensor where the last element of the input tensor is replaced with a randomly chosen element from the input tensor.

The `get_images_no_answer` function takes in an input tensor, an index and last_index and returns a tensor with last element removed from the input tensor.

The `repeat_last` function takes in an input tensor, an index and last_index and returns a tensor where the last element of the input tensor is repeated.


Example usage:

    # Get images based on indices
    inputs = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = tf.constant([[1], [2], [0]])
    output = get_images((inputs, index), 2)
    # output will be [[1, 2, 4], [4, 5, 7], [7, 8, 1]]

    # Get images with specific index
    inputs = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = get_images_with_index(inputs, index),index=1, last_index=2)
    # output will be [[1, 2, 2], [4, 5, 5], [7, 8, 8]]

    # Get images with random index
    inputs = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = random_last(inputs, index), last_index=2)
    # output will be [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Get images without answer
    inputs = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = get_images_no_answer(inputs, index), last_index=2)
    # output will be [[1, 2], [4, 5], [7, 8]]

    # Get images by repeating last
    inputs = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = repeat_last(inputs, index), last_index=2)
    # output will be [[1, 2, 2], [4, 5, 5], [7, 8, 8]]

"""
from typing import Tuple

import tensorflow as tf

from core_tools import ops as K


def get_matrix(inputs: tf.Tensor, index: tf.Tensor, last_index: int = 8) -> tf.Tensor:
    """
    Returns a tensor of shape (batch_size, last_index+1) which is a concatenation of inputs[:, :last_index] and K.gather(inputs, index[:, 0])[:, None]
    :param inputs : a tensor of shape (batch_size, features)
    :param index : a tensor of shape (batch_size, 1) representing the index of the element to be selected from inputs
    :param last_index : an int representing the index from where to split the input tensor. Default value is 8
    """
    return tf.concat([inputs[:, :last_index], K.gather(inputs, index[:, 0])[:, None]], axis=1)


def get_images(inputs: Tuple[tf.Tensor, tf.Tensor], last_index: int = 8) -> tf.Tensor:
    """
    Get images based on indices in the input tensor.
    :param inputs: Tuple containing the input tensor and indices tensor
    :param last_index: Index of last element to be included in the output tensor
    :return: Output tensor
    """
    return get_matrix(inputs[0], inputs[1], last_index=last_index)


def get_images_with_index(inputs: Tuple[tf.Tensor, tf.Tensor], index: int = 0, last_index: int = 8) -> tf.Tensor:
    """
    Get images from input tensor along with specific index.
    :param inputs: Tuple containing the input tensor and indices tensor
    :param index: Index of the element to be included in the output tensor
    :param last_index: Index of last element to be included in the output tensor
    :return: Output tensor
    """

    return tf.concat([inputs[0][:, :last_index], inputs[0][:, index:index + 1]], axis=1)


def random_last(inputs: Tuple[tf.Tensor, tf.Tensor], last_index: int = 8) -> tf.Tensor:
    """
    Get images from input tensor along with a randomly chosen index.
    :param inputs: Tuple containing the input tensor and indices tensor
    :param last_index: Index of last element to be included in the output tensor
    :return: Output tensor
    """
    index = K.init.label(max=last_index, shape=[tf.shape(inputs[0])[0]])[..., None]
    return get_matrix(inputs[0], index)


def get_images_no_answer(inputs: Tuple[tf.Tensor, tf.Tensor], last_index: int = 8) -> tf.Tensor:
    """
    Get images from input tensor without including any answer.
    :param inputs: Tuple containing the input tensor and indices tensor
    :param last_index: Index of last element to be included in the output tensor
    :return: Output tensor
    """
    return inputs[0][:, :last_index + 1]


def repeat_last(inputs: Tuple[tf.Tensor, tf.Tensor], last_index: int = 8) -> tf.Tensor:
    """
    Get images from input tensor by repeating the last element.
    :param inputs: Tuple containing the input tensor and indices tensor
    :param last_index: Index of last element to be included in the output tensor
    :return: Output tensor
    """
    return inputs[0][:, list(range(last_index)) + [last_index - 1]]
