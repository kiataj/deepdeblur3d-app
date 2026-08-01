import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from deblur3d_app.core import (
    DEFAULT_PRESET,
    TILE_PRESETS,
    normalize_float01,
    read_volume_auto,
    write_volume,
)


class NormalizeTests(unittest.TestCase):
    def test_already_normalized_float_is_not_copied(self):
        # The GUI normalizes for display and run_inference normalizes again;
        # the second call must not pay a full-volume copy.
        a = np.random.default_rng(0).random((4, 5, 6), dtype=np.float32)
        self.assertIs(normalize_float01(a), a)

    def test_integer_input_scales_by_dtype_max(self):
        a = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        out = normalize_float01(a)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0, places=4)

    def test_out_of_range_float_is_percentile_remapped(self):
        a = (np.random.default_rng(0).random((8, 8, 8)) * 5000 - 1000).astype(np.float32)
        out = normalize_float01(a)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_output_is_always_float32(self):
        # numpy 2's NEP 50 promotion turned the percentile branch's output into
        # float64, doubling the memory of every volume that took it.
        for dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
            for scale in (1.0, 100.0, 65535.0):
                with self.subTest(dtype=np.dtype(dtype).name, scale=scale):
                    a = (np.random.default_rng(0).random((4, 8, 8)) * scale).astype(dtype)
                    self.assertEqual(normalize_float01(a).dtype, np.float32)

    def test_two_dimensional_input_is_promoted(self):
        self.assertEqual(normalize_float01(np.zeros((4, 4), np.float32)).shape, (1, 4, 4))

    def test_rejects_four_dimensional_input(self):
        with self.assertRaisesRegex(ValueError, "Expected 3D or 2D"):
            normalize_float01(np.zeros((2, 2, 2, 2), np.float32))

    def test_degenerate_volume_does_not_divide_by_zero(self):
        out = normalize_float01(np.full((4, 4, 4), 7.0, dtype=np.float32) * 1000)
        self.assertTrue(np.all(np.isfinite(out)))


class PresetTests(unittest.TestCase):
    def test_default_preset_exists(self):
        self.assertIn(DEFAULT_PRESET, TILE_PRESETS)

    def test_default_preset_matches_historical_defaults(self):
        # Changing tile or overlap changes the tiling grid and therefore the
        # result; the default preset must stay bit-compatible with older runs.
        self.assertEqual(TILE_PRESETS[DEFAULT_PRESET], ((64, 256, 256), (32, 128, 128)))

    def test_every_overlap_is_smaller_than_its_tile(self):
        for name, (tile, overlap) in TILE_PRESETS.items():
            with self.subTest(preset=name):
                for axis, t, o in zip("ZYX", tile, overlap):
                    self.assertLess(o, t, f"{name} {axis}")
                    self.assertGreaterEqual(o, 0)

    def test_every_tile_meets_the_model_minimum(self):
        for name, (tile, _overlap) in TILE_PRESETS.items():
            with self.subTest(preset=name):
                self.assertTrue(all(t >= 16 for t in tile))


class VolumeIOTests(unittest.TestCase):
    def test_tiff_roundtrip_preserves_shape(self):
        vol = np.random.default_rng(0).random((6, 8, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.tif"
            write_volume(path, vol, dtype="uint16")
            self.assertTrue(path.is_file())
            self.assertEqual(read_volume_auto(path).shape, vol.shape)

    def test_npy_roundtrip_keeps_float32(self):
        vol = np.random.default_rng(0).random((4, 4, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.npy"
            write_volume(path, vol)
            np.testing.assert_allclose(read_volume_auto(path), vol, atol=1e-6)

    def test_reads_a_directory_of_slices_as_one_volume(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                tifffile.imwrite(str(Path(d) / f"s{i:03d}.tif"), np.full((8, 8), i * 1000, np.uint16))
            self.assertEqual(read_volume_auto(Path(d)).shape, (5, 8, 8))

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nested" / "deeper" / "out.tif"
            write_volume(path, np.zeros((4, 4, 4), np.float32))
            self.assertTrue(path.is_file())

    def test_rejects_unsupported_extension(self):
        with self.assertRaisesRegex(ValueError, "Unsupported input"):
            read_volume_auto(Path("scan.dcm"))


if __name__ == "__main__":
    unittest.main()
