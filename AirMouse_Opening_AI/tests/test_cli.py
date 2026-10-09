"""Dependency-free checks for safe local-file CLI behavior."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import airmouse


class LocalCliTests(unittest.TestCase):
    def test_missing_model_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            root = Path(folder)
            code = airmouse.main(['predict', '--model', str(root/'absent.pt'), '--source', str(root), '--output', str(root/'out')])
            self.assertEqual(code, 2)
            self.assertFalse((root/'out').exists())

    def test_directory_filters_assets_and_is_nonrecursive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'a.png').touch(); (root/'readme.txt').touch(); (root/'nested').mkdir()
            (root/'nested/b.jpg').touch()
            self.assertEqual([p.name for p in airmouse.input_images(str(root))], ['a.png'])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            root=Path(folder); (root/'model.pt').touch(); (root/'image.png').touch()
            out=root/'out';out.mkdir();sentinel=out/'keep.txt';sentinel.write_text('preserve')
            code=airmouse.main(['predict','--model',str(root/'model.pt'),'--source',str(root/'image.png'),'--output',str(out)])
            self.assertEqual(code,2);self.assertEqual(sentinel.read_text(),'preserve')

    def test_empty_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):airmouse.input_images(folder)


if __name__ == '__main__': unittest.main()
