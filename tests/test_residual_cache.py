import unittest

import numpy as np

from deblur3d_app.core import TILE_PRESETS, _cache_key


BASE = dict(
    tile=(64, 256, 256), overlap=(32, 128, 128), pad_mode="reflect",
    clamp01=True, use_amp="auto", batch_size="auto", border_margin="auto",
)


def key(vol, **overrides):
    return _cache_key("w.safetensors", "v2.0.0", "cuda", vol, **{**BASE, **overrides})


class CacheKeyTests(unittest.TestCase):
    def setUp(self):
        self.vol = np.random.default_rng(0).random((8, 16, 16), dtype=np.float32)

    def test_identical_settings_hit(self):
        self.assertEqual(key(self.vol), key(self.vol))

    def test_switching_preset_misses(self):
        # The reported bug: running "Low memory" then "Fast" reused the first
        # residual, which was computed on a different tiling grid.
        keys = set()
        for tile, overlap in TILE_PRESETS.values():
            keys.add(key(self.vol, tile=tile, overlap=overlap))
        self.assertEqual(len(keys), len(TILE_PRESETS))

    def test_every_inference_setting_changes_the_key(self):
        for field, other in [
            ("tile", (32, 128, 128)),
            ("overlap", (8, 32, 32)),
            ("pad_mode", "replicate"),
            ("clamp01", False),
            ("use_amp", False),
            ("batch_size", 4),
            ("border_margin", 0),
        ]:
            with self.subTest(field=field):
                self.assertNotEqual(key(self.vol), key(self.vol, **{field: other}))

    def test_a_different_volume_misses(self):
        other = np.random.default_rng(1).random((8, 16, 16), dtype=np.float32)
        self.assertNotEqual(key(self.vol), key(other))

    def test_device_and_weights_are_still_part_of_the_key(self):
        a = _cache_key("w.safetensors", "v2.0.0", "cuda", self.vol, **BASE)
        b = _cache_key("w.safetensors", "v2.0.0", "cpu", self.vol, **BASE)
        c = _cache_key("other.safetensors", "v2.0.0", "cuda", self.vol, **BASE)
        d = _cache_key("w.safetensors", "v1.0.0", "cuda", self.vol, **BASE)
        self.assertEqual(len({a, b, c, d}), 4)

    def test_key_is_order_independent(self):
        left = _cache_key("w", "v1", "cuda", self.vol, tile=(1, 2, 3), overlap=(4, 5, 6))
        right = _cache_key("w", "v1", "cuda", self.vol, overlap=(4, 5, 6), tile=(1, 2, 3))
        self.assertEqual(left, right)

    def test_key_is_hashable(self):
        hash(key(self.vol))


if __name__ == "__main__":
    unittest.main()
