# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

**Automatic batch sizing.** Tiles were processed one per forward pass, which
underutilizes the GPU now that memory is bounded by the tile rather than the
volume. `deblur_volume_tiled` takes a `batch_size` argument, defaulting to
`"auto"`, exposed in the widget as "Tiles per pass".

The budget is measured, not guessed. One probe forward at 32x64x64 gives the
activation slope for the loaded model, which is linear in tile voxels; free VRAM
comes from `torch.cuda.mem_get_info`. The probe costs about 10 ms, because the
2.5 s of CUDA context and cuDNN initialization it triggers is a one-time cost
the first real tile would otherwise pay. Two details matter: the probe subtracts
the baseline allocation, since the 35 MB of weights is a fixed cost that would
otherwise inflate the slope by 39% at probe size, and the result is scaled by
0.8, because extrapolation under-predicts by up to 12% on tile shapes that are
not powers of two. If a batch still does not fit, the batch is halved and
retried rather than failing the run.

Chosen batch sizes on an 8 GB card: 16 for a 32x64x64 tile, 6 for 64x128x128,
1 for the 64x256x256 default. CPU inference always uses 1.

Measured end to end, a 96x512x512 volume at a 32x64x64 tile goes from 5.6 s to
4.2 s (1.32x). Larger tiles gain much less, 1.02x at 32x128x128, because host
transfer and accumulation rather than the model dominate there. The gain is
concentrated in small tiles, which is the regime memory-constrained users are in.

Batching does not change the tiling grid, so Hann blending is untouched and
`batch_size=1` is bit-for-bit identical to the previous implementation. Wider
batches differ by about 1e-5 (max 1.24e-5 measured) because cuDNN may select a
different kernel per batch size; `cudnn.deterministic = True` does not remove
this. Set "Tiles per pass" to 1 for bit-reproducible output.

The chosen batch size is recorded in `layer.metadata["deblur3d"]`.

### Not doing

**Automatic tile sizing from free VRAM** was designed and measured, then
dropped. Tile size determines the tiling grid, so changing it changes the
result: against a 32x64x64 reference the trained model differs by a mean of
1.3e-2 and a max of 2.0e-1, roughly 850 levels of a 16-bit range. Selecting the
tile from available VRAM would make output depend on the GPU and on whatever
else was using it at the time, which is not acceptable for quantitative work.
Tile size stays an explicit user choice.

## v2.0.0

### Fixed

**Out-of-memory on large volumes.** Reported by a user processing a large
dataset. GPU memory scaled with the size of the *volume*, not the size of the
tile, so tiling did not bound memory the way it was supposed to. Three places
allocated whole-volume tensors on the inference device:

| Location | Tensors | Cost |
|---|---|---|
| `deblur_volume_tiled` | input volume, `out`, `wei` | 12 bytes/voxel |
| control step in `run_infer_bound` | `vol_t`, `base_t`, `r_t`, `y_ctrl` | 16 bytes/voxel |
| `gaussian_blur3d`, when `hp_sigma > 0` | blur temporaries | +8 bytes/voxel |
| `_RES_CACHE` | `vol_t`, `r_t` per entry | 8 bytes/voxel, never released |

On top of that, per-tile activations measure a linear 708 bytes/voxel, so the
default tile of 64x256x256 alone needs 2.97 GB. A 1024^3 volume therefore asked
for roughly 12.9 GB of accumulators before the model had run a single tile.

Now the volume and the blending accumulators stay in host memory and only the
current tile is resident on the device; the control step is applied in Z-slabs
with a halo matching the Gaussian kernel radius. Peak GPU memory measured on an
RTX 3060 Ti at a fixed 32x128x128 tile:

| Volume | Voxels | Before | After |
|---|---|---|---|
| 64x256x256 | 4.2 M | 0.46 GB | 0.41 GB |
| 128x256x256 | 8.4 M | 0.51 GB | 0.41 GB |
| 256x512x512 | 67.1 M | 1.21 GB | 0.41 GB |
| 384x768x768 | 226.5 M | 3.13 GB | 0.41 GB |

GPU memory is now independent of volume size and bounded by the tile alone, so
tile size is the single knob that controls it. Output is bit-for-bit identical
to the previous implementation: loop and accumulation order are unchanged, and
slab-wise control application was verified against whole-volume application for
`hp_sigma` of 0.0, 1.5 and 4.0.

Note for anyone who saw slowness rather than a crash: on Windows the NVIDIA
driver silently spills oversubscribed allocations to system RAM, so the same
bug surfaced as severe slowdown instead of an error.

**Residual cache pinned VRAM for the whole session.** `_RES_CACHE` held two
whole-volume device tensors per entry and was never evicted, so every volume
processed in a session permanently reserved 8 bytes/voxel of VRAM. Entries are
now host-resident and the cache is capped at one entry.

**CPU fallback retried while the failed CUDA allocation was still held.** The
device cache is now released before falling back.

### Added

**Automatic versioning from git tags.** The version was a hand-edited `0.1.0`
in `pyproject.toml` and the repository had no tags, so a released build could
not be traced back to a commit. setuptools-scm now derives the version at build
time; running from a checkout or an editable install falls back to
`git describe`, because an editable install freezes its recorded version at
install time and goes stale as commits land. Both paths normalize to PEP 440
using the same guess-next-dev scheme, so a tagged commit is the bare tag
(`2.0.0`) and later commits sort above it (`2.0.1.dev7+gdeadbee`). To cut a
release, tag the commit; nothing is edited by hand.

**Result provenance.** Output layers now record the app version, the model
repository, revision, `model_version` and `arch_version`, the device, and the
tile, overlap and control parameters in `layer.metadata["deblur3d"]`. The app
version is printed at startup and shown in the viewer title. Model weights were
already versioned through HuggingFace semver tags, but nothing tied a given
result to the code that produced it.

### Changed

**Build artifacts are no longer tracked in git.** `build/`,
`src/deblur3d_gui.egg-info/` and the `__pycache__` `.pyc` files were committed
to the repository. Because building regenerates them, the working tree went
dirty mid-build and setuptools-scm stamped release wheels `2.0.1.dev0+d<date>`
instead of the tag they were built from. They are removed from the index, kept
on disk, and covered by a new `.gitignore`.

### Known limitations

- Tiles are processed one per forward pass. Batching them for throughput is
  designed but not implemented.
- The default overlap of 128 on a 256 tile is 50%, producing 343 tiles where an
  overlap of 32 would produce 125. Reducing it is roughly a 2.7x speedup but
  changes output, so the default is unchanged.
- The volume is normalized twice on the host, once in the GUI and again in
  `run_infer_bound`, and `np.percentile` copies internally. This is a separate
  host-memory path that has not been addressed.
