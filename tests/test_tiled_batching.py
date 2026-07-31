import unittest

import numpy as np
import torch

from deblur3d.infer.tiled import _auto_batch_size, deblur_volume_tiled


class BatchRecordingNet(torch.nn.Module):
    """Identity net that records the batch size of every forward it sees."""

    def __init__(self, levels=4):
        super().__init__()
        self.down = [None] * levels
        self.batches = []

    def forward(self, value):
        self.batches.append(int(value.shape[0]))
        return value * 0.5


class OOMOnceNet(BatchRecordingNet):
    """Raises a CUDA OOM for any batch wider than `limit`, to drive the backoff."""

    def __init__(self, limit, levels=4):
        super().__init__(levels=levels)
        self.limit = limit

    def forward(self, value):
        if value.shape[0] > self.limit:
            self.batches.append(int(value.shape[0]))
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().forward(value)


class BatchingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.vol = rng.random((32, 64, 64), dtype=np.float32)
        self.tile = (16, 32, 32)
        self.overlap = (4, 8, 8)

    def _run(self, net, batch_size):
        return deblur_volume_tiled(
            net, self.vol, tile=self.tile, overlap=self.overlap,
            device="cpu", batch_size=batch_size,
        )

    def test_batching_matches_single_tile_output(self):
        # With a batch-invariant net, blending must be untouched by batch width.
        reference = self._run(BatchRecordingNet(), 1)
        for batch_size in (2, 3, 4, 8):
            with self.subTest(batch_size=batch_size):
                got = self._run(BatchRecordingNet(), batch_size)
                np.testing.assert_array_equal(reference, got)

    def test_batch_size_is_honoured(self):
        net = BatchRecordingNet()
        self._run(net, 4)
        self.assertTrue(all(b <= 4 for b in net.batches), net.batches)
        self.assertIn(4, net.batches)

    def test_every_tile_is_still_processed(self):
        one, many = BatchRecordingNet(), BatchRecordingNet()
        self._run(one, 1)
        self._run(many, 4)
        self.assertEqual(sum(one.batches), sum(many.batches))

    def test_backs_off_on_out_of_memory_instead_of_failing(self):
        net = OOMOnceNet(limit=2)
        got = self._run(net, 8)
        self.assertEqual(got.shape, self.vol.shape)
        self.assertTrue(max(net.batches) <= 8)
        # It must actually have retried at a narrower width, not just given up.
        self.assertTrue(any(b <= 2 for b in net.batches), net.batches)

    def test_backoff_result_matches_unbatched(self):
        reference = self._run(BatchRecordingNet(), 1)
        got = self._run(OOMOnceNet(limit=2), 8)
        np.testing.assert_array_equal(reference, got)

    def test_non_oom_runtime_error_propagates(self):
        class Broken(BatchRecordingNet):
            def forward(self, value):
                raise RuntimeError("shape mismatch")

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            self._run(Broken(), 4)

    def test_rejects_unknown_batch_size_string(self):
        with self.assertRaisesRegex(ValueError, "must be an int or 'auto'"):
            self._run(BatchRecordingNet(), "bogus")

    def test_auto_falls_back_to_one_on_cpu(self):
        self.assertEqual(
            _auto_batch_size(BatchRecordingNet(), torch.device("cpu"), 1000, 16), 1
        )

    def test_auto_is_accepted_on_cpu(self):
        reference = self._run(BatchRecordingNet(), 1)
        np.testing.assert_array_equal(reference, self._run(BatchRecordingNet(), "auto"))


if __name__ == "__main__":
    unittest.main()
