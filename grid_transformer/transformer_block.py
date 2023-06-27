"""
This module contains two classes `TransformerBlockBase` and `TransformerBlock`, which implement the transformer block as defined in the "Attention Is All You Need" paper by Google.
A transformer block consists of a multi-head self-attention mechanism and a position-wise feed-forward neural network.

Classes:

- TransformerBlockBase:  A transformer block with original implementation of the call function
- TransformerBlock: A transformer block with modified implementation of the call function which changes the order of layernorm and residual connection.


Example usage:

    import tensorflow as tf

    # Instantiate the transformer block
    tb = transformer_block.TransformerBlockBase()

    # Build the transformer block with input shape (batch_size, sequence_length, input_dim)
    tb.build((None, 10, 32))

    # Perform the forward pass with input tensor of shape (batch_size, sequence_length, input_dim)
    output = tb(tf.random.normal((32, 10, 32)))

    # Perform the forward pass and return attention scores with input tensor of shape (batch_size, sequence_length, input_dim)
    output, scores = tb(tf.random.normal((32, 10, 32)), return_attention_scores=True)

    # Instantiate a transformer block with user-specified feed-forward network
    ffn = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(32)])
    tb = transformer_block.TransformerBlock(ffn=ffn)

    # Instantiate a TransformerBlock2
    tb2 = transformer_block.TransformerBlock()

    # Build the transformer block with input shape (batch_size, sequence_length, input_dim)
    tb2.build((None, 10, 32))

    # Perform the forward pass with input tensor of shape (batch_size, sequence_length, input_dim)
    output = tb2(tf.random.normal((32, 10, 32)))

    # Perform the forward pass and return attention scores with input tensor of shape (batch_size, sequence_length, input_dim)
    output, scores = tb2(tf.random.normal((32, 10, 32)), return_attention_scores=True)

    # Instantiate a transformer block with user-specified feed-forward network
    ffn = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(32)])
    tb = transformer_block.TransformerBlock(ffn=ffn)
"""
from typing import Union, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, Dropout, LayerNormalization


class TransformerBlockBase(Layer):
    """
    A transformer block as defined in the "Attention Is All You Need" paper by Google.
    It consists of a multi-head self-attention mechanism and a position-wise feed-forward neural network.
    """

    def __init__(self, num_heads=8, size=None, ff_mul=4, ff_size=None, dropout=0.1, mha_dropout=0.1, ff_act="gelu",
                 ffn=None, return_attention_scores=False):
        """
        Initialize the transformer block.

        :param num_heads: Number of heads in the multi-head self-attention mechanism
        :param size: Dimension of the input/output vectors. If None, it will be inferred from input_shape during build()
        :param ff_mul: Multiplier for the feed-forward network size
        :param ff_size: Size of the feed-forward network. If None, it will be inferred from size and ff_mul
        :param dropout: Dropout rate for the feed-forward network
        :param mha_dropout: Dropout rate for the multi-head self-attention mechanism
        :param ff_act: Activation function for the feed-forward network
        :param ffn: A user-specified feed-forward network. If None, a default one will be created
        :param return_attention_scores: Whether to return attention scores
        """
        super(TransformerBlockBase, self).__init__()
        self.size = size
        self.num_heads = num_heads
        self.ff_mul = ff_mul
        self.ff_size = ff_size
        self.ff_act = ff_act
        self.dropout = dropout
        self.mha_dropout = mha_dropout
        self.return_attention_scores = return_attention_scores
        self.ffn = ffn

    def build(self, query_shape, value_shape=None, key_shape=None):
        """
        Build the transformer block.

        :param query_shape: Shape of the input tensor
        """
        if self.size is None:
            self.size = query_shape[-1]
        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.size, dropout=self.mha_dropout, )
        if self.ff_size is None:
            self.ff_size = self.size * self.ff_mul
        # self.ffn = self.ffn or build_dense_model(
        #     [
        #         self.ff_size,
        #         Dropout(self.dropout),
        #         self.size,
        #         Dropout(self.dropout)
        #     ],
        #     activation=self.ff_act,
        #     last_activation=None)
        self.ffn = self.ffn or Sequential([
            Dense(self.ff_size, activation=self.ff_act),
            # Dropout(self.dropout),
            Dense(self.size),
            Dropout(self.dropout)
        ])
        self.dropout = Dropout(self.dropout)
        self.layernorm_1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(
            self,
            query: tf.Tensor,
            value: tf.Tensor = None,
            key: tf.Tensor = None,
            attention_mask: tf.Tensor = None,
            return_attention_scores: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Perform the forward pass of the transformer block.
        :param query: Query tensor
        :param value: Value tensor
        :param key: Key tensor
        :param attention_mask: Attention mask tensor
        :param return_attention_scores: Whether to return attention scores
        :return: Output tensor and attention scores if return_attention_scores is True
        """
        x = query
        if value is None:
            value = x
        return_attention_scores = self.return_attention_scores if return_attention_scores is None else return_attention_scores
        if return_attention_scores:
            x, scores = self.att(
                query=x,
                value=value,
                key=key,
                return_attention_scores=True
            )
        else:
            x = self.att(
                query=x,
                value=value,
                key=key,
            )
        x = self.layernorm_1(x + query)
        y = self.ffn(x)
        if return_attention_scores:
            return self.layernorm2(x + y), scores
        return self.layernorm2(x + y)

    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size,
            "num_heads": self.num_heads,
            "ff_mul": self.ff_mul,
            "ff_size": self.ff_size,
            "ff_act": self.ff_act,
            "dropout": self.dropout,
            "mha_dropout": self.mha_dropout,
            "return_attention_scores": self.return_attention_scores,
            "ffn": self.ffn
        })
        return config


class TransformerBlock(TransformerBlockBase):
    """

    A
    TransformerBlock
    with a modified call function, which changes the order of layernorm and residual connection.
    """

    def call(
            self,
            query: tf.Tensor,
            value: tf.Tensor=None,
            key: tf.Tensor=None,
            attention_mask: tf.Tensor=None,
            return_attention_scores: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Perform the forward pass of the transformer block.
        :param query: Query tensor
        :param value: Value tensor
        :param key: Key tensor
        :param attention_mask: Attention mask tensor
        :param return_attention_scores: Whether to return attention scores
        :return: Output tensor and attention scores if return_attention_scores is True
        """
        x = self.layernorm_1(query)
        if value is None:
            value = x

        return_attention_scores = self.return_attention_scores if return_attention_scores is None else return_attention_scores

        if return_attention_scores:
            x, scores = self.att(
                query=x,
                value=value,
                key=key,
                return_attention_scores=True
            )
        else:
            x = self.att(
                query=x,
                value=value,
                key=key,
            )
        x = query + self.dropout(x)
        # x = query + x
        y = self.layernorm2(x)
        y = self.ffn(y)
        if return_attention_scores:
            return x + y, scores
        return x + y


