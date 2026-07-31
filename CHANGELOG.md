# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

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

- Tile size is still chosen by hand. Automatic tile selection from available
  VRAM, plus batching tiles per forward pass for throughput, is designed but
  not implemented.
- The default overlap of 128 on a 256 tile is 50%, producing 343 tiles where an
  overlap of 32 would produce 125. Reducing it is roughly a 2.7x speedup but
  changes output, so the default is unchanged.
- The volume is normalized twice on the host, once in the GUI and again in
  `run_infer_bound`, and `np.percentile` copies internally. This is a separate
  host-memory path that has not been addressed.
