import unittest

import numpy as np
import torch

from deblur3d.infer.tiled import _border_margin, _starts, deblur_volume_tiled


class IdentityNet(torch.nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.down = [None] * levels

    def forward(self, value):
        return value


def weight_map(length, tile, overlap, margin):
    """Blending weight each voxel along one axis receives."""
    window = torch.hann_window(tile, periodic=False).clamp_min(1e-6).numpy()
    total = np.zeros(length)
    for start in _starts(length, tile, overlap, margin):
        lo, hi = max(0, start), min(length, start + tile)
        total[lo:hi] += window[lo - start: lo - start + (hi - lo)]
    return total


class StartsTests(unittest.TestCase):
    def test_zero_margin_reproduces_the_original_grid(self):
        for length, tile, overlap in [(80, 32, 8), (200, 64, 32), (512, 64, 32)]:
            with self.subTest(length=length):
                starts = _starts(length, tile, overlap, 0)
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1], length - tile)

    def test_margin_extends_past_both_ends(self):
        starts = _starts(200, 64, 32, 16)
        self.assertEqual(starts[0], -16)
        self.assertEqual(starts[-1], 200 + 16 - 64)

    def test_grid_still_covers_the_whole_volume(self):
        for margin in (0, 4, 16):
            with self.subTest(margin=margin):
                covered = np.zeros(200, bool)
                for s in _starts(200, 64, 32, margin):
                    covered[max(0, s): min(200, s + 64)] = True
                self.assertTrue(covered.all())


class BorderMarginTests(unittest.TestCase):
    def test_border_weight_was_negligible_without_a_margin(self):
        # The bug: outermost voxels were a single tile's edge prediction at a
        # blending weight of 1e-6 against ~1 inside.
        w = weight_map(200, 64, 32, margin=0)
        self.assertLess(w[0], 1e-5)
        self.assertLess(w[-1], 1e-5)
        self.assertGreater(np.median(w[64:-64]), 0.5)

    def test_margin_restores_border_weight(self):
        w = weight_map(200, 64, 32, margin=_border_margin((64, 64, 64), (32, 32, 32), 16)[0])
        interior = np.median(w[64:-64])
        self.assertGreater(w[0], 0.1 * interior)
        self.assertGreater(w[-1], 0.1 * interior)

    def test_no_voxel_is_left_unweighted(self):
        w = weight_map(200, 64, 32, margin=16)
        self.assertGreater(w.min(), 1e-4)

    def test_margin_is_zero_when_there_is_no_overlap(self):
        # Without overlap there is no blending window to speak of.
        self.assertEqual(_border_margin((64, 64, 64), (0, 0, 0), 16), (0, 0, 0))

    def test_margin_is_capped_by_tile_and_overlap(self):
        self.assertEqual(_border_margin((16, 16, 16), (8, 8, 8), 16), (4, 4, 4))
        self.assertEqual(_border_margin((256, 256, 256), (128, 128, 128), 16), (16, 16, 16))


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.vol = np.linspace(0, 1, 24 * 40 * 40, dtype=np.float32).reshape(24, 40, 40)

    def _run(self, **kw):
        return deblur_volume_tiled(
            IdentityNet(), self.vol, tile=(16, 16, 16), overlap=(4, 4, 4),
            device="cpu", **kw,
        )

    def test_identity_round_trips_with_the_margin(self):
        np.testing.assert_allclose(self._run(), self.vol, rtol=1e-5, atol=1e-6)

    def test_identity_round_trips_without_the_margin(self):
        np.testing.assert_allclose(self._run(border_margin=0), self.vol, rtol=1e-5, atol=1e-6)

    def test_explicit_margin_is_accepted_as_int_and_tuple(self):
        np.testing.assert_allclose(self._run(border_margin=4), self.vol, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            self._run(border_margin=(4, 2, 2)), self.vol, rtol=1e-5, atol=1e-6
        )

    def test_rejects_unknown_margin_string(self):
        with self.assertRaisesRegex(ValueError, "border_margin must be"):
            self._run(border_margin="lots")

    def test_margin_adds_no_tiles(self):
        # The outermost tiles are shifted, not added, so the fix is free.
        def count(**kw):
            seen = []
            deblur_volume_tiled(
                IdentityNet(), self.vol, tile=(16, 16, 16), overlap=(4, 4, 4),
                device="cpu", progress=lambda d, t: seen.append(t), **kw,
            )
            return seen[-1]

        self.assertEqual(count(), count(border_margin=0))


class ShiftDoesNotOpenGapsTests(unittest.TestCase):
    def test_coverage_holds_across_many_geometries(self):
        for length in (40, 80, 97, 200, 512):
            for tile, overlap in ((16, 4), (32, 8), (64, 32), (64, 16)):
                if tile > length:
                    continue
                margin = _border_margin((tile,) * 3, (overlap,) * 3, 16)[0]
                with self.subTest(length=length, tile=tile, overlap=overlap):
                    covered = np.zeros(length, bool)
                    for s in _starts(length, tile, overlap, margin):
                        covered[max(0, s): min(length, s + tile)] = True
                    self.assertTrue(covered.all(), "gap in coverage")

    def test_tile_count_is_unchanged_unless_one_tile_spans_the_axis(self):
        for length, tile, overlap in [(80, 32, 8), (200, 64, 32), (512, 64, 32), (97, 32, 8)]:
            margin = _border_margin((tile,) * 3, (overlap,) * 3, 16)[0]
            with self.subTest(length=length, tile=tile):
                self.assertEqual(
                    len(_starts(length, tile, overlap, margin)),
                    len(_starts(length, tile, overlap, 0)),
                )

    def test_single_tile_axis_needs_a_second_tile(self):
        self.assertEqual(len(_starts(16, 16, 4, 0)), 1)
        self.assertEqual(len(_starts(16, 16, 4, 4)), 2)


if __name__ == "__main__":
    unittest.main()
