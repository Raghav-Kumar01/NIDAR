"""Regression checks for source clipping and preserving existing annotations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from continue_annotations import clip, save


class ContinuationTests(unittest.TestCase):
    def test_clipping_preserves_sloped_intersections(self):
        polygon = clip([[-1., 0.], [1., 0.], [1., 1.]])
        self.assertEqual({tuple(p) for p in polygon},
                         {(0., .5), (0., 0.), (1., 0.), (1., 1.)})

    def test_fully_outside_polygon_is_empty(self):
        self.assertEqual(clip([[2., 0.], [3., 0.], [3., 1.]]), [])

    def test_existing_review_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'reviews').mkdir()
            (root/'reviewed_labels').mkdir()
            path = root/'reviews/example.json'
            original = b'{"method":"explicit_browser_review"}'
            path.write_bytes(original)
            self.assertFalse(save(root, {'id': 'example'}, 'negative', [],
                                  'ai_visual_review', 'test', 'test'))
            self.assertEqual(path.read_bytes(), original)

    def test_changed_image_cannot_be_annotated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'reviews').mkdir()
            (root/'reviewed_labels').mkdir()
            image = root/'image.bin'
            image.write_bytes(b'changed')
            rec = {'id': 'example', 'image': str(image),
                   'sha256': hashlib.sha256(b'original').hexdigest()}
            with self.assertRaises(ValueError):
                save(root, rec, 'negative', [], 'ai_visual_review', 'test', 'test')
            self.assertFalse((root/'reviews/example.json').exists())


if __name__ == '__main__':
    unittest.main()
