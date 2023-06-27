import unittest

import tensorflow as tf

from grid_transformer.layers import Detokenizer, Vec


class TestConversion(unittest.TestCase):
    def test_conversion(self):
        size = 9
        max_ = None
        input_shape = (10, 100, 512)
        model = Detokenizer(size, max_)
        model.build(input_shape)
        inputs = tf.random.normal((10, 100, 512))
        output = model(inputs)
        self.assertEqual(output.shape, (10, 9, 576))

    def test_conversion_with_max(self):
        size = 9
        max_ = 50
        input_shape = (10, 100, 512)
        model = Detokenizer(size, max_)
        model.build(input_shape)
        inputs = tf.random.normal((10, 100, 512))
        output = model(inputs)
        self.assertEqual(output.shape, (10, 9, 288))

    def test_conversion_with_invalid_input_shape(self):
        size = 9
        max_ = None
        input_shape = (10, 100, 513)
        model = Detokenizer(size, max_)
        with self.assertRaises(ValueError):
            model.build(input_shape)


class TestVec(unittest.TestCase):
    def test_vec(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(2)])
        vec_model = Vec(model)
        x = tf.random.normal((2, 3, 4, 5, 6))
        expected_output_shape = (3, 4, 5, 6, 2)
        output = vec_model(x)
        self.assertEqual(output.shape, expected_output_shape)

    def test_vec_with_different_input_shape(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(2)])
        vec_model = Vec(model)
        x = tf.random.normal((1, 5, 4, 3, 2))
        expected_output_shape = (5, 4, 3, 2, 2)
        output = vec_model(x)
        self.assertEqual(output.shape, expected_output_shape)

    def test_vec_with_different_model(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(20, activation='sigmoid'), tf.keras.layers.Dense(3)])
        vec_model = Vec(model)
        x = tf.random.normal((2, 3, 4, 5, 6))
        expected_output_shape = (3, 4, 5, 6, 3)
        output = vec_model(x)
        self.assertEqual(output.shape, expected_output_shape)


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
