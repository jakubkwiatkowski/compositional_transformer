import numpy as np
import tensorflow as tf

from grid_transformer import Transformer


def test_transformer():
    inputs = (tf.random.normal((5, 10, 20)), tf.random.uniform((5, 1), minval=0, maxval=19, dtype=tf.int32))
    apply_transformer = Transformer()
    output = apply_transformer(inputs)
    assert output.shape == (5, 1000), f'Incorrect output shape. Expected (5, 1000) but got {output.shape}'

def test_transformer_attention_scores():
    inputs = (tf.random.normal((5, 10, 20)), tf.random.uniform((5, 1), minval=0, maxval=19, dtype=tf.int32))
    apply_transformer = Transformer(return_attention_scores=True)
    output, attention_scores = apply_transformer(inputs)
    assert output.shape == (5, 1000), f'Incorrect output shape. Expected (5, 1000) but got {output.shape}'
    assert len(attention_scores) == 8, f'Incorrect number of attention scores. Expected 8 but got {len(attention_scores)}'
    assert attention_scores[0].shape == (5,10,10), f'Incorrect attention score shape. Expected (5,10,10) but got {attention_scores[0].shape}'

def test_transformer_pos_emd():
    inputs = (tf.random.normal((5, 10, 20)), tf.random.uniform((5, 1), minval=0, maxval=19, dtype=tf.int32))
    apply_transformer = Transformer(pos_emd='sum')
    output = apply_transformer(inputs)
    assert output.shape == (5, 1000), f'Incorrect output shape. Expected (5, 1000) but got {output.shape}'

def test_transformer_extractor():
    inputs = (tf.random.normal((5, 10, 20)), tf.random.uniform((5, 1), minval=0, maxval=19, dtype=tf.int32))
    apply_transformer = Transformer(extractor='resnet50', extractor_pooling='max')
    output = apply_transformer(inputs)
    assert output.shape == (5, 1000), f'Incorrect output shape. Expected (5, 1000) but got {output.shape}'
