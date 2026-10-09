"""Run with python -B -m unittest discover -s annotation_tools."""
import unittest
from full_dataset import review_payload, validate, yolo_text


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.poly = [[0, 0], [1, 0], [1, 1]]

    def test_valid_polygon(self):
        validate([self.poly])

    def test_bad_coordinates(self):
        for value in (-0.1, 1.1, float('nan'), float('inf'), True, '0.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate([[[0, 0], [value, 0], [1, 1]]])

    def test_degenerate_polygon(self):
        with self.assertRaises(ValueError):
            validate([[[0, 0], [.5, .5], [1, 1]]])

    def test_incomplete_polygon(self):
        with self.assertRaises(ValueError):
            validate([[[0, 0], [1, 1]]])

    def test_empty_positive_cannot_be_approved(self):
        with self.assertRaises(ValueError):
            review_payload('positive', [])

    def test_negative_cannot_keep_a_polygon(self):
        with self.assertRaises(ValueError):
            review_payload('negative', [self.poly])

    def test_empty_negative_is_explicit(self):
        review_payload('negative', [])
        self.assertEqual(yolo_text([]), '')

    def test_unknown_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            review_payload('unknown', [])

    def test_valid_positive_serializes_class_zero(self):
        review_payload('positive', [self.poly])
        self.assertTrue(yolo_text([self.poly]).startswith('0 '))


if __name__ == '__main__':
    unittest.main()
