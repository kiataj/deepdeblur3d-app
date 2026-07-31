import unittest

import numpy as np
import torch

from deblur3d.infer.tiled import (
    deblur_volume_tiled,
    validate_tiling,
    validate_volume_shape,
)


class RecordingIdentityNet(torch.nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.down = [None] * levels
        self.seen_shapes = []

    def forward(self, value):
        self.seen_shapes.append(tuple(value.shape[2:]))
        return value


class VolumeShapeValidationTests(unittest.TestCase):
    def test_accepts_minimum_volume_shape(self):
        self.assertEqual(validate_volume_shape((16, 16, 16)), (16, 16, 16))

    def test_rejects_thin_z_stack_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, r"Z=8.*at least 16 slices"):
            validate_volume_shape((8, 512, 512))

    def test_rejects_non_3d_input_with_stack_hint(self):
        with self.assertRaisesRegex(ValueError, "Open Files as Stack"):
            validate_volume_shape((512, 512))

    def test_accepts_minimum_tile_size(self):
        self.assertEqual(
            validate_tiling((16, 256, 256), (8, 128, 128)),
            ((16, 256, 256), (8, 128, 128)),
        )

    def test_rejects_tile_below_model_minimum(self):
        with self.assertRaisesRegex(ValueError, r"at least 16.*Z=8"):
            validate_tiling((8, 256, 256), (4, 128, 128))

    def test_rejects_overlap_equal_to_tile(self):
        with self.assertRaisesRegex(ValueError, r"overlap=16, tile=16"):
            validate_tiling((16, 256, 256), (16, 128, 128))

    def test_rejects_negative_overlap(self):
        with self.assertRaisesRegex(ValueError, r"overlap=-1"):
            validate_tiling((16, 256, 256), (-1, 128, 128))

    def test_default_sized_tile_is_capped_to_minimum_volume(self):
        volume = np.linspace(0.0, 1.0, 16**3, dtype=np.float32).reshape(16, 16, 16)
        net = RecordingIdentityNet()

        result = deblur_volume_tiled(
            net,
            volume,
            tile=(64, 64, 64),
            overlap=(32, 32, 32),
            device="cpu",
        )

        # Every tile is capped to the volume. There is more than one because the
        # grid is extended past each end so border voxels are not tile-edge
        # predictions; see BorderMarginTests.
        self.assertTrue(net.seen_shapes)
        self.assertTrue(all(s == (16, 16, 16) for s in net.seen_shapes), net.seen_shapes)
        np.testing.assert_allclose(result, volume, rtol=1e-5, atol=1e-6)

    def test_zero_overlap_preserves_tile_boundaries(self):
        volume = np.linspace(0.0, 1.0, 16 * 16 * 32, dtype=np.float32).reshape(
            16, 16, 32
        )

        result = deblur_volume_tiled(
            RecordingIdentityNet(),
            volume,
            tile=(16, 16, 16),
            overlap=(0, 0, 0),
            device="cpu",
        )

        np.testing.assert_allclose(result, volume, rtol=1e-5, atol=1e-6)

    def test_volume_minimum_tracks_model_depth(self):
        volume = np.zeros((16, 32, 32), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, r"at least 32.*Z=16"):
            deblur_volume_tiled(
                RecordingIdentityNet(levels=5),
                volume,
                tile=(64, 64, 64),
                overlap=(32, 32, 32),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
