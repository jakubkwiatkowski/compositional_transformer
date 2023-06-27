"""
The module contains two classes, `Conversion` and `Vec`.

`Conversion` class is a model that converts input sequence of tokens to a sequence of tokens desired size and desired number of tokens. It can be initialized with a desired size and maximum token number. The build method is used to set the token_max and mul attributes based on the input_shape, and the call method performs the conversion by reshaping the inputs tensor.

`Vec` class is a model that applies a given model on each element of the input tensor. It is initialized with a model, and the call method applies the model on each element of the input tensor by first transposing the input tensor and then using tf.vectorized_map to apply the model on each element, and then transposing the output tensor.

Example usage:
```python
# create an instance of Conversion model
conversion_model = Conversion(size=5, max_=20)
conversion_model.build((None, 30, 100))

# create an instance of Vec model
vec_model = Vec(tf.keras.layers.Dense(units=64))

# input tensor
x = tf.random.normal((10,30,100))

# apply conversion model on input tensor
converted_x = conversion_model(x)

# apply vec model on input tensor
vec_x = vec_model(x)
```
"""

from typing import Tuple, Optional

import tensorflow as tf
import math

from loguru import logger
from tensorflow.keras import Model


class Detokenizer(Model):
    """
    A model for converting input sequence of tokens to a sequence of tokens desired size and desired number of tokens..
    """

    def __init__(self, size: int = 9, max_: Optional[int] = None, max_output: Optional[int] = 1280):
        """
        Initialize the class with desired size and maximum token number.

        :param size: The desired size of the output tokens.
        :param max_: The maximum number of tokens in the output shape.
        """
        super().__init__()
        self.token_max = max_  # number off transformer output token that will be used to create model output
        self.output_size_max = max_output
        self.output_no = size  # number of outputs
        self.mul = 1

    def build(self, input_shape: Tuple[int, int, int]) -> None:

        """
        Build the model.

        :param input_shape: The input shape of the model.
        """
        if self.output_size_max:
            full_output = self.output_size_max * self.output_no
            full_size = input_shape[1] * input_shape[2]
            if full_size > full_output:
                ratio = full_size / full_output
                self.token_max = math.floor(input_shape[1] / ratio)

        if self.token_max is None:
            ratio = input_shape[1] / self.output_no
            if ratio < 1:
                self.mul = math.ceil(1 / ratio)
                self.token_max = self.output_no
                if input_shape[2] % self.mul != 0:
                    # todo high rise error
                    logger.error(
                        f"To create {self.output_no} from {input_shape[1]} tokens the size of transformer {input_shape[2]} need to divisible by {self.mul}.")

            else:
                self.token_max = math.floor(ratio) * self.output_no

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the conversion.

        :param inputs: Input tensor of shape (batch_size, tokens, features).
        :return: Tensor of shape (batch_size, size, features * (token_max / size)).
        """
        shape = tf.shape(inputs)
        if self.mul > 1:
            inputs = tf.reshape(inputs, (shape[0], int(inputs.shape[1] * self.mul), int(inputs.shape[2] / self.mul)))
            shape = tf.shape(inputs)
        return tf.reshape(inputs[:, :self.token_max],
                          tf.stack(
                              [shape[0], self.output_no, int(inputs.shape[-1] * (self.token_max / self.output_no))]))


class Vec(tf.keras.Model):
    def __init__(self, model: tf.keras.Model):
        """
        Class that applies a given model on each element of the input tensor.

        Parameters:
            model (tf.keras.Model): The model to apply on each element of the input tensor.
        """
        super().__init__()
        self.model = model

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the model on each element of the input tensor.

        Parameters:
            x (tf.Tensor): The input tensor

        Returns:
            tf.Tensor: The output tensor with the model applied on each element.
        """
        x = tf.transpose(x, perm=(1, 0, 2, 3, 4))
        x = tf.vectorized_map(self.model, x)
        return tf.transpose(x, perm=(1, 0, 2, 3, 4))
