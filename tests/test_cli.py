import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tifffile

from deblur3d_app.cli import _collect_inputs, _parse_triplet, build_parser, main


class ParserTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["in.tif", "out.tif"])
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.dtype, "uint16")
        self.assertEqual(args.tiles_per_pass, "auto")
        self.assertFalse(args.batch)
        self.assertFalse(args.no_amp)

    def test_control_parameters_are_parsed(self):
        args = build_parser().parse_args(
            ["a.tif", "b.tif", "--strength", "1.5", "--hp-sigma", "2.0",
             "--hp-gain", "0.5", "--lp-gain", "1.25"]
        )
        self.assertAlmostEqual(args.strength, 1.5)
        self.assertAlmostEqual(args.hp_sigma, 2.0)
        self.assertAlmostEqual(args.hp_gain, 0.5)
        self.assertAlmostEqual(args.lp_gain, 1.25)

    def test_tile_override_takes_three_values(self):
        args = build_parser().parse_args(["a.tif", "b.tif", "--tile", "32", "64", "64"])
        self.assertEqual(args.tile, [32, 64, 64])

    def test_rejects_unknown_preset(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["a.tif", "b.tif", "--preset", "nonexistent"])

    def test_list_presets_exits_cleanly_without_io_args(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--list-presets"]), 0)
        self.assertIn("Balanced (default)", buf.getvalue())


class TripletTests(unittest.TestCase):
    def test_none_passes_through(self):
        self.assertIsNone(_parse_triplet(None, "tile"))

    def test_three_values_become_a_tuple(self):
        self.assertEqual(_parse_triplet([1, 2, 3], "tile"), (1, 2, 3))

    def test_wrong_arity_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_triplet([1, 2], "tile")


class CollectInputsTests(unittest.TestCase):
    def test_finds_volume_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("a.tif", "b.tiff", "c.npy"):
                (root / name).write_bytes(b"")
            (root / "notes.txt").write_bytes(b"")
            found = {p.name for p in _collect_inputs(root, "*")}
            self.assertEqual(found, {"a.tif", "b.tiff", "c.npy"})

    def test_falls_back_to_slice_directories(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for vol in ("scan1", "scan2"):
                (root / vol).mkdir()
                tifffile.imwrite(str(root / vol / "s0.tif"), np.zeros((4, 4), np.uint16))
            found = {p.name for p in _collect_inputs(root, "*")}
            self.assertEqual(found, {"scan1", "scan2"})

    def test_glob_narrows_the_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("keep_1.tif", "keep_2.tif", "skip.tif"):
                (root / name).write_bytes(b"")
            found = {p.name for p in _collect_inputs(root, "keep_*")}
            self.assertEqual(found, {"keep_1.tif", "keep_2.tif"})

    def test_empty_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                _collect_inputs(Path(d), "*")

    def test_file_instead_of_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.tif"
            f.write_bytes(b"")
            with self.assertRaises(SystemExit):
                _collect_inputs(f, "*")


if __name__ == "__main__":
    unittest.main()
