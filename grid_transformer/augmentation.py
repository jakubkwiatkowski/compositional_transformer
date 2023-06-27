"""

This module contains a collection of functions and classes for working with tensors in TensorFlow.

Functions:

- `shuffle_fn : A wrapper function for shuffle layer that allows for customization of the axis along which to shuffle.
- `shuffle`: A callable function that applies the shuffle operation on a given tensor with default axis 0.
- `shuffle_col`: A callable function that applies the shuffle operation on a given tensor with axis 1.
- `transpose`: A partial function that returns a transpose layer, which can be used to transpose the dimensions of a tensor.
- `reshape_static`: A function to reshape a tensor in a specific shape and apply a given function to it.

Classes:

- `Transpose(Layer)`: Returns a transpose layer, which can be used to transpose the dimensions of a tensor.
- `PartialModel(Layer)`: Returns a PartialModel layer that allows for applying a function to a subset of the input tensor.

Example usage:

    import tensorflow as tf
    from models_utils import reshape_static, transpose

    # Create a tensor of shape (2,3)
    x = tf.constant([[1,2,3], [4,5,6]])

    # Reshape the tensor to shape (2, 2, 3) and apply transpose function
    y = reshape_static([2, 2, 3], transpose)(x)
    print(y)

    # Output:
    # [[[1 3]
    #   [2 4]]
    #  [[5 6]
    #   [3 2]]]


    import numpy as np
    from models_utils import PartialModel

    # Create a tensor of shape (2, 2, 3)
    x = np.random.rand(2, 2, 3)

    # Create a partial model layer with indices [0, 1] and function sum
    pm = PartialModel(fn=tf.reduce_sum, indices=[0, 1])

    # Apply the partial model layer on input tensor
    y = pm(x)
    print(y)

    # Output:
    # [[1.7268854  1.80438087 1.35692811]
    #  [1.74081294 1.70722026 1.45006824]]

In this example, the function `tf.reduce_sum` is applied to the first two indices of the input tensor `x` and the resulting tensor is returned.
"""
from functools import partial
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Lambda
from core_tools import ops as K
import math
from typing import List, Union, Callable, Optional, Tuple

from core_tools.core import il, lw
from core_tools.ops.mask import random_update


def shuffle_fn(axis: int = 0) -> Callable:
    """
    A wrapper function for Keras's shuffle layer that allows for customization of the axis along which to shuffle.

    :param axis: The axis along which to shuffle. Default is 0.
    :type axis: int
    :return: A callable function that applies the shuffle operation on a given tensor.
    :rtype: Callable
    """
    return K.vec(K.shuffle, axis=axis)


shuffle = shuffle_fn()
shuffle_col = shuffle_fn(axis=1)
transpose = partial(tf.transpose, perm=(0, 2, 1, 3, 4))


def reshape_static(shape: Union[List[int], tf.TensorShape],
                   model: Callable = shuffle,
                   row: Union[None, int] = None) -> Callable:
    """
    A function to reshape a tensor in a specific shape and apply a given function to it.

    :param shape: The shape of the reshaped tensor. It can be a list of integers or a tf.TensorShape object.
    :type shape: Union[List[int], tf.TensorShape]
    :param model: A callable function to apply on the reshaped tensor. Default is the shuffle function.
    :type model: Callable
    :param row: If provided, number of row in reshaped tensor, use to calculate the reshape shape.
    :type row: Union[None, int]
    :return: A callable function that reshape the tensor and apply the given function.
    :rtype: Callable
    """
    if not il(shape):
        shape = tuple(shape.shape)
    if row is None:
        row = math.floor(math.sqrt(shape[1]))
    newshape = [-1] + [row, row] + list(shape[2:])
    shape = (-1,) + shape[1:]

    def fn(x):
        x = tf.reshape(x, newshape)
        x = model(x)
        x = tf.reshape(x, shape)
        return x

    return fn


class Transpose(Layer):
    """
    This class returns a transpose layer, which can be used to transpose the dimensions of a tensor.
    """

    def __init__(self, axis: tuple = (1, 0)):
        """
        Initialize the Transpose layer

        :param axis: The axis along which to transpose the tensor. Defaults to (1, 0).
        :type axis: tuple
        """
        super().__init__()
        self.axis = axis

    def build(self, input_shape: tuple) -> None:
        """
        Constructs the transpose layer, it will take input_shape as input and it will not return any output.
        :param input_shape: Tensor shape of the input.
        :type input_shape: tuple
        """
        dim = len(input_shape)
        axis_len = len(self.axis)
        if axis_len < dim:
            self.axis = tuple(self.axis) + tuple(range(dim))[axis_len:]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        This function will take inputs tensor as input and it will return the transposed tensor.

        :param inputs: Input tensor to be transposed.
        :type inputs: tf.Tensor
        :return: Transposed tensor.
        :rtype: tf.Tensor
        """
        return tf.transpose(inputs, self.axis)


class PartialModel(Layer):
    """
    This class returns a PartialModel layer, which can be used to apply a model on a subset of the input tensor.
    """

    def __init__(self, model: Model, last_axis: int = 8):
        """
        Initialize the PartialModel layer

        :param model: The model to be applied to the subset of the input tensor
        :type model: Model
        :param last_axis: The last axis on which the model should be applied. Defaults to 8.
        :type last_axis: int
        """
        super().__init__()
        self.model = model
        self.last_axis = last_axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        This function will take inputs tensor as input and it will return the combined output of model and the original tensor.

        :param inputs: Input tensor to be processed.
        :type inputs: tf.Tensor
        :return: combined output tensor of model and the original tensor.
        :rtype: tf.Tensor
        """
        x = self.model(inputs[:, :self.last_axis])
        return tf.concat([x, inputs[:, self.last_axis:]], axis=1)


def noise_seq(inputs: tf.Tensor, row: Optional[int] = None, col: Optional[int] = None,
              sample_from: str = "sample") -> tf.Tensor:
    """
    This function adds noise to a sequence by updating some of its elements.
    :param inputs: the input sequence to be manipulated.
    :type inputs: tf.Tensor
    :param row: the number of rows of the input sequence. If not provided, it defaults to the second dimension of the input tensor shape.
    :type row: int
    :param col: the number of columns of the input sequence. If not provided, it defaults to the third dimension of the input tensor shape.
    :type col: int
    :param sample_from: a string that specifies the noise source. "sample" (default) will sample from the input sequence while "batch" will sample from the batch.
    :type sample_from: str
    :return: a tensor with the same shape as the input but with some of its elements updated.
    :rtype: tf.Tensor
    """
    shape = tf.shape(inputs)
    row = row if row else shape[1]
    col = col if col else shape[2]
    batch = shape[0]
    indexes = K.tensor([
        tf.range(batch),
        K.init.label(max=row, shape=batch[None]),
        K.init.label(max=col, shape=batch[None])
    ]).T
    updates = ([
        K.init.label(max=batch, shape=batch[None]) if sample_from == "batch" else tf.range(batch),
        K.init.label(max=row, shape=batch[None]),
        K.init.label(max=col, shape=batch[None])
    ])

    return tf.tensor_scatter_nd_update(
        tensor=inputs,
        indices=indexes,
        updates=inputs[updates]
    )


class Noise(Layer):
    def __init__(self, last_index: Optional[int] = None, sample_from: str = "sample", prob: float = 1.0):
        """
        Initialize the Noise layer
        :param last_index: the last index of the input sequence. If not provided, it defaults to the second dimension of the input tensor shape.
        :type last_index: int
        :param sample_from: a string that specifies the noise source. "sample" (default) will sample from the input sequence while "batch" will sample from the batch.
        :type sample_from: str
        :param prob: a float value between 0 and 1 that represents the probability of applying noise to the input.
        :type prob: float
        """
        super().__init__()
        self.last_index = last_index
        self.sample_from = sample_from
        self.prob = prob

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Constructs the noise layer, it will take input_shape as input and it will not return any output.
        :param input_shape: Tensor shape of the input.
        :type input_shape: tuple
        """
        if not self.last_index:
            self.last_index = input_shape[1]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        This function will take inputs tensor as input and it will return the noise tensor.
        :param inputs: Input tensor to be manipulated.
        :type inputs: tf.Tensor
        :return: Tensor with noise added.
        :rtype: tf.Tensor
        """
        shape = tf.shape(inputs)
        switch_indexes = tf.range(shape[0])
        if self.prob < 1.0:
            switch_indexes = switch_indexes[K.categorical_like(inputs, prob=self.prob, dtype="bool")[0]]
            switch_no = tf.shape(switch_indexes)[0]
        else:
            switch_no = shape[0]
        indexes = tf.transpose(K.tensor([
            switch_indexes,
            K.init.label(max=self.last_index, shape=switch_no[None]),
        ]), perm=(1, 0))
        updates = ([
            K.init.label(max=switch_no, shape=switch_no[None]) if self.sample_from == "batch" else tf.range(
                switch_no),
            K.init.label(max=self.last_index, shape=switch_no[None]), ])

        return tf.tensor_scatter_nd_update(
            tensor=inputs,
            indices=indexes,
            updates=inputs[updates]
        )


BatchNoise = partial(Noise, sample_from="batch")


def noise(inputs: tf.Tensor, last_index: Optional[int] = None, sample_from: str = "sample",
          prob: float = 1.0) -> tf.Tensor:
    """
    This function will take inputs tensor as input and it will return the noise tensor.
    :param inputs: Input tensor to be manipulated.
    :type inputs: tf.Tensor
    :param last_index: the last index of the input sequence. If not provided, it defaults to the second dimension of the input tensor shape.
    :type last_index: int
    :param sample_from: a string that specifies the noise source. "sample" (default) will sample from the input sequence while "batch" will sample from the batch.
    :type sample_from: str
    :param prob: a float value between 0 and 1 that represents the probability of applying noise to the input.
    :type prob: float
    :return: Tensor with noise added.
    :rtype: tf.Tensor
    """
    shape = tf.shape(inputs)
    last_index = last_index if last_index else shape[1]
    switch_indexes = tf.range(shape[0])
    if prob < 1.0:
        switch_indexes = switch_indexes[K.categorical_like(inputs, prob=prob, dtype="bool")[0]]
        switch_no = tf.shape(switch_indexes)[0]
    else:
        switch_no = shape[0]
    indexes = tf.transpose(K.tensor([
        switch_indexes,
        K.init.label(max=last_index, shape=switch_no[None]),
    ]), perm=(1, 0))
    updates = ([
        K.init.label(max=switch_no, shape=switch_no[None]) if sample_from == "batch" else tf.range(switch_no),
        K.init.label(max=last_index, shape=switch_no[None]),
    ])

    return tf.tensor_scatter_nd_update(
        tensor=inputs,
        indices=indexes,
        updates=inputs[updates]
    )




def rand(augmentation: Union[float, int, List[Union[float, Tuple[float, callable]]], Tuple[float, callable]],
         root: int = 2, transpose: bool = True) -> Callable:
    """
    This function will take augmentation,root and transpose as input and it will return a callable function
    :param augmentation: a list of floats or tuples of floats and callable functions that represents the probability of applying noise to the input.
    :type augmentation: Union[float, int, List[Union[float, Tuple[float, callable]]], Tuple[float, callable]]
    :param root: an integer that represents the root of the number of the augmentation functions.
    :type root: int
    :param transpose: a boolean value that represents whether the input should be transposed or not.
    :type transpose: bool
    :return: callable function
    :rtype: callable
    """
    augmentation_fn = [
        (lambda x=m: K.list_.DivideDim(x, root=root)) for m in (
                [shuffle_fn(i) for i in range(root)]
                + ([Transpose((0, 2, 1))] if transpose else [])
        )
    ]
    if isinstance(augmentation, (float, int)):
        augmentation = [augmentation] * len(augmentation_fn)
    aug = []
    for i, a in enumerate(lw(augmentation)):
        if len(lw(a)) == 2:
            aug.append(a)
        else:
            aug.append((a, augmentation_fn[i]()))

    def apply(*args):
        indexes = Lambda(K.range_like)(args[0])
        # aug_indexes = K.range_like(inputs)
        for k, v in aug:
            if k > 0:
                indexes = random_update(indexes, model=v, prob=k)

        outputs = [K.gather(a, indexes) for a in args]
        return tuple(outputs)

    return apply


def rand_aug4(*args, prob: Optional[float] = None, vec: Optional[bool] = True) -> List:
    """
    Apply random augmentation to a list of inputs with the specified probability.
    :param args: list of inputs to be augmented
    :param prob: probability of applying augmentation to each input.
                If None, it will be set to 1/len(args)
    :param vec: boolean, whether to use vectorized version of layers or not
    :return: list of augmented inputs
    """
    layers = [partial(K.shuffle, axis=0), K.shuffle, partial(tf.transpose, perm=(1, 0))]
    return K.rand.switch(*args, layers=layers, prob=prob, vec=vec)


def rand_aug2(*args, prob: Optional[float] = None, vec: Optional[Callable] = K.list_.run) -> List:
    """
    Apply random augmentation to a list of inputs with the specified probability.
    :param args: list of inputs to be augmented
    :param prob: probability of applying augmentation to each input.
                If None, it will be set to 1/len(args)
    :param vec: function to use for vectorizing the input list, defaults to `K.list_.run`
    :return: list of augmented inputs
    """
    layers = [partial(K.list_.shuffle, axis=0), K.list_.shuffle, partial(K.list_.transpose)]
    return K.rand.switch(*args, layers=layers, prob=prob, vec=vec)


def rand_aug3(*args, prob: Optional[float] = None, vec: Optional[Callable] = K.list_.run) -> List:
    """
    Apply random augmentation to a list of inputs with the specified probability.
    :param args: list of inputs to be augmented
    :param prob: probability of applying augmentation to each input.
                If None, it will be set to 1/len(args)
    :param vec: function to use for vectorizing the input list, defaults to `K.list_.run`
    :return: list of augmented inputs
    """
    layers = [DivideShape(f, axis=0, replace_first=False) for f in
              [partial(K.list_.shuffle, axis=0), K.list_.shuffle, partial(K.list_.transpose)]]
    return K.rand.switch(*args, layers=layers, prob=prob, vec=vec)


batch_noise_seq = partial(noise_seq, sample_from='batch')
batch_noise = partial(noise, sample_from='batch')
