import numpy as np
import unittest

from grid_transformer.preprocessing import get_images, get_images_no_answer, repeat_last, get_matrix


class TestGetMatrix(unittest.TestCase):
    def test_get_matrix(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])
        last_index = 3

        expected_output = np.array([[1, 2, 3, 3], [6, 7, 8, 9]])
        output = get_matrix(inputs, index, last_index)

        np.testing.assert_array_equal(output, expected_output)

    def test_get_matrix_with_default_last_index(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])

        expected_output = np.array([[1, 2, 3, 3, 4], [6, 7, 8, 9, 10]])
        output = get_matrix(inputs, index)

        np.testing.assert_array_equal(output, expected_output)

    def test_get_matrix_with_invalid_inputs(self):
        inputs = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
        index = np.array([[2], [5]])
        last_index = 3

        self.assertRaises(IndexError, get_matrix, inputs, index, last_index)

    def test_get_matrix_with_different_batch_sizes(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3], [2]])
        last_index = 3

        self.assertRaises(ValueError, get_matrix, inputs, index, last_index)


class TestGetImages(unittest.TestCase):
    def test_get_images(self):
        inputs = (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), np.array([[2], [3]]))
        index = np.array([[2], [3]])
        last_index = 3

        expected_output = np.array([[1, 2, 3, 3], [6, 7, 8, 9]])
        output = get_images((inputs, index), last_index)

        np.testing.assert_array_equal(output, expected_output)

    def test_get_images_with_default_last_index(self):
        inputs = (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), np.array([[2], [3]]))
        index = np.array([[2], [3]])

        expected_output = np.array([[1, 2, 3, 3, 4], [6, 7, 8, 9, 10]])
        output = get_images((inputs, index))

        np.testing.assert_array_equal(output, expected_output)

    def test_get_images_with_invalid_inputs(self):
        inputs = (np.array([[1, 2, 3, 4], [6, 7, 8, 9]]), np.array([[2], [5]]))
        index = np.array([[2], [3]])
        last_index = 3

        self.assertRaises(IndexError, get_images, inputs, last_index)

    def test_get_images_with_different_batch_sizes(self):
        inputs = (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), np.array([[2], [3], [2]]))
        index = np.array([[2], [3]])
        last_index = 3

        self.assertRaises(ValueError, get_images, inputs, last_index)


class TestGetImagesNoAnswer(unittest.TestCase):
    def test_get_images_no_answer(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])
        last_index = 3

        expected_output = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
        output = get_images_no_answer((inputs, index), last_index)

        np.testing.assert_array_equal(output, expected_output)

    def test_get_images_no_answer_with_default_last_index(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])

        expected_output = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        output = get_images_no_answer((inputs, index))

        np.testing.assert_array_equal(output, expected_output)


class TestRepeatLast(unittest.TestCase):
    def test_repeat_last(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])
        last_index = 4

        expected_output = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
        output = repeat_last((inputs, index), last_index)

        np.testing.assert_array_equal(output, expected_output)

    def test_repeat_last_with_default_last_index(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        index = np.array([[2], [3]])

        expected_output = np.array([[1, 2, 3, 4, 4], [6, 7, 8, 9, 9]])
        output = repeat_last((inputs, index))

        np.testing.assert_array_equal(output, expected_output)


if __name__ == '__main__':
    unittest.main()
