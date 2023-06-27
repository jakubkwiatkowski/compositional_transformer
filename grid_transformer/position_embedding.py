"""
The  module contains models for adding positional embeddings to tokens.

The `TokenAndPositionEmbedding` class is a Keras layer that concatenates token embeddings with position embeddings. The layer takes in three parameters - `maxlen` which is the maximum length of input sequences, `vocab_size` which is the vocabulary size of the input sequences and `embed_dim` which is the dimension of the embeddings. The layer has a call method that takes in an input tensor `x` with shape (batch_size, sequence_length) and returns a tensor with shape (batch_size, sequence_length, embed_dim). In the call method, it uses the token_emb Embedding layer to embed the input tensor and pos_emb Embedding layer to embed the positions. Finally, it adds the position embeddings to the token embeddings and returns the concatenated tensor.

The `CatPositionEmbedding` class is a Keras layer that concatenates position embeddings to the input tensor. The layer takes in a single parameter `embed_dim` which is the dimension of the position embeddings. The layer has a build method that takes in an input shape tuple (batch_size, sequence_length) and creates an Embedding layer pos_emb with input_dim as the second element of the input shape tuple and output_dim as the embed_dim. The layer also has a call method that takes in an input tensor `x` with shape (batch_size, sequence_length) and returns a tensor with shape (batch_size, sequence_length, embed_dim). In the call method, it uses the pos_emb Embedding layer to embed the positions and concatenates it with the input tensor along the last axis.

The `SumPositionEmbedding` class is a Keras layer that sums position embeddings to the input tensor. It has a build method that takes in an input shape tuple (batch_size, sequence_length, embed_dim) and creates an Embedding layer pos_emb with input_dim as the second element of the input shape tuple and output_dim as the third element of the input shape tuple. The layer also has a call method that takes in an input tensor `x` with shape (batch_size, sequence_length, embed_dim) and returns a tensor with shape (batch_size, sequence_length, embed_dim). In the call method, it uses the pos_emb Embedding layer to embed the positions and adds it to the input tensor.
"""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
class TokenAndPositionEmbedding(Layer):
    """
    A Keras Layer that concatenates token embeddings with position embeddings.
    :param maxlen: The maximum length of input sequences.
    :param vocab_size: The vocabulary size of the input sequences.
    :param embed_dim: The dimension of the embeddings.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        :param x: Input tensor with shape (batch_size, sequence_length)
        :return: Tensor with shape (batch_size, sequence_length, embed_dim)
        """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class CatPositionEmbedding(Layer):
    """
    A Keras Layer that concatenates position embeddings to the input tensor.
    :param embed_dim: The dimension of the position embeddings.
    """
    def __init__(self, embed_dim: int=16):
        super().__init__()
        self.embed_dim = embed_dim

    def build(self, input_shape: Tuple[int, int]):
        """
        Builds the layer
        :param input_shape: The shape of the input, in the format (batch_size, sequence_length)
        """
        self.pos_emb = Embedding(input_dim=input_shape[1], output_dim=self.embed_dim)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        :param x: Input tensor with shape (batch_size, sequence_length)
        :return: Tensor with shape (batch_size, sequence_length, embed_dim)
        """
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_emb(positions)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [tf.shape(x)[0], 1, 1])
        return tf.concat([x, positions], axis=-1)


class SumPositionEmbedding(Layer):
    """
    A Keras Layer that sums position embeddings to the input tensor.
    """
    def build(self, input_shape: Tuple[int, int, int]):
        """
        Builds the layer
        :param input_shape: The shape of the input, in the format (batch_size, sequence_length, embed_dim)
        """
        self.pos_emb = Embedding(input_dim=input_shape[1], output_dim=input_shape[2])
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        :param x: Input tensor with shape (batch_size, sequence_length, embed_dim)
        :return: Tensor with shape (batch_size, sequence_length, embed_dim)
        """
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_emb(positions)
        return x + positions
