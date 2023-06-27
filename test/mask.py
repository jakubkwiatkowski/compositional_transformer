import unittest
import numpy as np
import tensorflow as tf
from grid_transformer.mask import empty_last, random_mask, constant_mask, mix, take_by_index, init_weights, ImageMask


class TestInitWeights(unittest.TestCase):
    def test_init_weights(self):
        # Test default dtype
        weights = init_weights((3, 3))
        self.assertIsInstance(weights, tf.Tensor)
        self.assertEqual(weights.dtype, tf.float32)

        # Test custom dtype
        weights = init_weights((3, 3), tf.float16)
        self.assertIsInstance(weights, tf.Tensor)
        self.assertEqual(weights.dtype, tf.float16)

        # Test shape
        weights = init_weights((3, 3))
        self.assertEqual(weights.shape, (3, 3))

        # Test if weights are initialized with glorot uniform distribution
        weights = init_weights((100, 100))
        self.assertTrue(np.all(weights >= -np.sqrt(6 / (100 + 100))))
        self.assertTrue(np.all(weights <= np.sqrt(6 / (100 + 100))))


class TestTakeByIndex(unittest.TestCase):
    def test_take_by_index(self):
        # Test array of shape (2, 3, 4) with default i
        x = np.random.rand(2, 3, 4)
        result = take_by_index(x)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, x[..., 8:9]))

        # Test array of shape (2, 3, 4) with custom i
        x = np.random.rand(2, 3, 4)
        result = take_by_index(x, i=2)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, x[..., 2:3]))

        # Test array of shape (2, 3, 4) with i > shape of array
        x = np.random.rand(2, 3, 4)
        result = take_by_index(x, i=4)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, x[..., 4:5]))

        # Test array of shape (2, 3, 4) with i < 0
        x = np.random.rand(2, 3, 4)
        result = take_by_index(x, i=-1)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, x[..., -1:]))


class TestMix(unittest.TestCase):
    def test_mix(self):
        # Test array of shape (2, 3, 4)
        x = np.random.rand(2, 3, 4)
        result = mix(x)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, (x[..., 7:8] + x[..., 5:6]) / 2))

        # Test array of shape (2, 3, 4)
        x = np.random.rand(2, 3, 4)
        result = mix(x)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, (x[..., 7:8] + x[..., 5:6]) / 2))

        # Test array of shape (2, 3, 4)
        x = np.random.rand(2, 3, 4)
        result = mix(x)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result, (x[..., 7:8] + x[..., 5:6]) / 2))


class TestEmptyLast(unittest.TestCase):
    def test_empty_last(self):
        # Test tensor of shape (2, 3, 4)
        x = tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32)
        result = empty_last(x)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertTrue(np.array_equal(result.numpy(), np.zeros((2, 3, 1))))

        # Test tensor of shape (4, 5, 6)
        x = tf.constant(np.random.rand(4, 5, 6), dtype=tf.float32)
        result = empty_last(x)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertTrue(np.array_equal(result.numpy(), np.zeros((4, 5, 1))))

        # Test tensor of shape (3, 2, 4)
        x = tf.constant(np.random.rand(3, 2, 4), dtype=tf.float32)
        result = empty_last(x)
        self.assertEqual(result.shape, (3, 2, 1))
        self.assertTrue(np.array_equal(result.numpy(), np.zeros((3, 2 , 1))))


class TestRandomMask(unittest.TestCase):
    def test_random_mask(self):
        # Test tensor of shape (2, 3, 4)
        inputs = tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32)
        result = random_mask(inputs, last_index=4)
        self.assertEqual(result.shape, inputs.shape)
        self.assertTrue(result.dtype == tf.bool)

        # Test tensor of shape (4, 5, 6)
        inputs = tf.constant(np.random.rand(4, 5, 6), dtype=tf.float32)
        result = random_mask(inputs, last_index=6)
        self.assertEqual(result.shape, inputs.shape)
        self.assertTrue(result.dtype == tf.bool)

        # Test tensor of shape (3, 2, 4)
        inputs = tf.constant(np.random.rand(3, 2, 4), dtype=tf.float32)
        result = random_mask(inputs, last_index=4)
        self.assertEqual(result.shape, inputs.shape)
        self.assertTrue(result.dtype == tf.bool)


class TestConstantMask(unittest.TestCase):
    def test_constant_mask(self):
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
        value = 2
        last_index = 5
        expected_output = tf.constant([[False, True, False], [False, False, False]])
        output = constant_mask(inputs, value, last_index)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_constant_mask_default_value(self):
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
        expected_output = tf.constant([[False, False, False], [False, False, False]])
        output = constant_mask(inputs)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_constant_mask_default_last_index(self):
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
        value = 8
        expected_output = tf.constant([[False, False, False], [False, False, False]])
        output = constant_mask(inputs, value)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))


class TestImageMask(unittest.TestCase):
    def setUp(self):
        self.model = ImageMask(lambda x: x[:, -1], last_index=None)

    def test_call(self):
        inputs = tf.ones((1, 2, 2, 3))
        mask = tf.ones((1, 2, 2, 1))
        output = self.model((inputs, mask))
        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (1, 2, 2, 3))

    def test_call_last_index(self):
        inputs = tf.ones((1, 2, 2, 3))
        mask = tf.ones((1, 2, 2, 1))
        self.model.last_index = 2
        output = self.model((inputs, mask))
        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (1, 2, 2, 2))

    def test_call_last_index_out_of_bounds(self):
        inputs = tf.ones((1, 2, 2, 3))
        mask = tf.ones((1, 2, 2, 1))
        self.model.last_index = 4
        self.assertRaises(ValueError, self.model, (inputs, mask))

    def test_call_wrong_input_shape(self):
        inputs = tf.ones((1, 2, 2))
        mask = tf.ones((1, 2, 2, 1))
        self.assertRaises(ValueError, self.model, (inputs, mask))


if __name__ == '__main__':
    unittest.main()
