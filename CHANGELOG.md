# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

**The outermost slice of each axis was an unblended tile-edge prediction.**
Reported as the last slice looking sharper and noisier than the rest. The first
and last slice, row and column were each covered by exactly one tile, at a Hann
blending weight of 1e-6 against about 0.98 in the interior. Since the accumulator
is normalized by that weight, those voxels were the raw prediction from the very
edge of a single tile, where the U-Net's zero-padded convolutions are least
reliable. Everywhere else that error is averaged away by an overlapping
neighbour; at the volume border there is no neighbour.

The last slice was worse than the first because the model's residual declines
smoothly toward the end of a tile (0.097, 0.089, 0.077, 0.071 on successive
slices) and then jumps back to 0.090 on the final one, leaving it over-corrected
relative to its neighbours.

The tile grid now nudges its outermost tiles past each end of the volume, with
the out-of-volume part reflected, so real voxels always sit inside a tile. On a
test volume cropped from the interior of a larger block, high-frequency energy in
the last slice relative to the interior went from 1.22 to 1.04, and in the first
slice from 0.83 to 1.03.

The outermost tiles are shifted rather than added, so this is close to free:
1.046s to 1.058s on a 96x384x384 volume. Only an axis spanned by a single tile
needs an extra one, which is 0.2% of the 10,432 geometries swept. That sweep also
confirms the shift never opens a gap in coverage; an earlier version of the fix
did, on volumes with only two tiles along an axis, which the sweep caught.

This changes output at the volume border. Pass `--legacy-borders` (CLI) or
`border_margin=0` (API) to reproduce earlier results.

**The border shift must not spend the overlap it borrows from.** The first cut of
the fix above capped the shift at the full overlap, so on the "Fast" preset
(16-voxel Z overlap on a 64-deep tile) the outermost tiles ended up merely
abutting their neighbour with zero overlap. That replaced the border artifact
with a worse unblended interior seam: 49.5% slice-to-slice deviation at the seam,
against 47.4% at the border it was fixing. Reported from the GUI as the same
artifact appearing on interior slices.

The shift is now capped at half the overlap, and the backoff treats a
zero-overlap seam as a failure rather than accepting it as gapless. Worst-case
deviation on that preset drops from 49.5% to 4.2%.

**The "Low memory" preset had too little Z overlap to blend at all.** Eight
voxels on a 32-deep tile is 25%, which leaves each seam blending two predictions
taken 4 voxels from a tile edge, well inside the contaminated zone: 16%
deviation. Raised to 16, matching the half-tile overlap the other presets use,
which brings it to 4.8%. It costs time, not memory, which is the right trade for
a preset chosen to fit in limited VRAM.

All three presets now sit in a 3.9-4.8% band, measured against the median slice
of a volume cropped from the interior of a larger block.

**The residual cache ignored the tiling settings.** Running one preset, then
switching to another, silently reused the residual computed with the previous
tiling grid: the key was only (weights, revision, device, volume fingerprint).
Reported from the GUI after switching from "Low memory" to "Fast". The two
presets differ by 2.0e-2 on the same volume, so the stale result was visibly
wrong, and the preset recorded in provenance did not match the pixels.

The key now covers everything that changes the network's output: tile, overlap,
pad mode, clamping, AMP, batch size and border margin. The control parameters
are still excluded, since reusing one residual across control sweeps is the
whole point of the cache. This predates the preset dropdown, which only made it
easy to trigger; changing the tile spin boxes by hand hit the same bug.

### Added

**An info badge beside the Tiling control.** Hovering the "i" explains what tile
size does and describes each preset with its VRAM cost and how it trades
blending against speed, so the choice is not guesswork. Each dropdown entry carries its own
tooltip too.

**Command line interface.** `deblur3d IN OUT` runs a single volume headlessly;
`deblur3d IN_DIR OUT_DIR --batch` processes a folder. Inputs may be TIFF stacks,
`.npy` arrays, or directories of TIFF slices. Existing outputs are skipped unless
`--overwrite` is given, so an interrupted batch resumes rather than redoing work.
`--list-presets` prints the tiling options. Batch runs never prompt for a model
upgrade, so a study cannot silently change model revisions halfway through.

This required splitting the non-GUI logic out of `gui.py` into `core.py`, which
imports neither napari nor Qt. Verified bit-for-bit identical to the previous
implementation on a full inference run.

**Progress reporting and an abort button.** `deblur_volume_tiled` accepts
`progress(done, total)` and `should_abort()` callbacks. The GUI gains a progress
panel showing the tile count with an Abort button; the CLI shows a tqdm bar, plus
an outer bar over volumes in batch mode. Aborting raises `InferenceAborted`,
which deliberately does not subclass `RuntimeError` so the tile loop's OOM retry
can never mistake it for an out-of-memory condition.

**Update notification for the app, in both front ends.** The GUI checks on
startup and offers to open the release page, and has a "Check for updates" button
for an on-demand check that also confirms when you are current. The CLI prints a
notice to stderr, suppressible with `--no-update-check` or the
`DEBLUR3D_NO_UPDATE_CHECK` environment variable. This mirrors the existing prompt
for new model revisions. The GUI check runs on a background thread, and both fail
silently offline.

**The control sliders show an editable value box.** The readout was already an
editable spin box, but superqt draws it frameless so it read as a static label
and users did not realise they could type an exact value. Strength, HP Sigma, HP
Gain and LP Gain now show a bordered field with spin arrows and a tooltip, under
an "Inference-time controls" heading.

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

### Changed

**Mixed precision is on by default, worth 1.87x.** `use_amp` defaulted to False
behind the comment "PT1.12 + InstanceNorm: keep False unless you use GroupNorm".
The model is GroupNorm throughout, so the condition the comment names as safe was
the one that actually held, and the default had been leaving most of the GPU's
throughput unused. `use_amp="auto"` now enables autocast whenever the device is
CUDA and the model contains no InstanceNorm, checked rather than assumed, since
the architecture is read from `config.json` and could change.

Measured 1.87x at a 32x64x64 tile and 1.74x at 32x128x128, end to end. Numerical
cost is about 6.5e-4, roughly 42 levels of a 16-bit range: larger than batching's
1e-5, far smaller than the 2e-1 that changing tile size costs. Pass `--no-amp`
(CLI) for bit-reproducibility against pre-2.1 runs.

`cudnn.benchmark` was evaluated at the same time and does nothing here, since
tile shapes are fixed and cuDNN's heuristics already pick the same kernels.

**The GUI is down from 16 controls to 6.** What remains is the documented
interface: device, tiling preset, and the four control parameters. Removed:
`clamp01` (the model already clamps internally, and unchecking it broke the
contrast slider's [0,1] assumption), `pad_mode` (edge-tile detail nobody tunes),
`use_amp` (now automatic), `reuse_cache` (the cache key already covers weights,
revision, device and volume fingerprint, so reuse is always correct), and the
"Clear residual cache" button (needed only when the cache was unbounded and
GPU-resident; it is now capped at one host-side entry). The six tile and overlap
spin boxes collapse into named presets, of which "Balanced (default)" reproduces
the previous defaults exactly. The active preset is recorded in provenance.

**The volume is normalized once instead of twice.** `normalize_float01` now
returns already-normalized float32 input untouched instead of paying a full
`np.clip` copy, so the GUI's display normalization and the inference path no
longer duplicate a full-volume allocation.

**The tile loop is pipelined, so the GPU stays fed.** GPU utilization sampled
during a run oscillated between 74% and 100%, averaging about 90%: the GPU idled
while the host packed the next batch, copied it over, and blended the previous
one. Two things forced that serialization. Pageable host memory makes every copy
synchronous, and the copy back ran once per tile, so a batch of 16 cost 16
separate synchronization points.

Tiles are now packed into pinned staging buffers and transferred on a dedicated
CUDA stream, the copy back is one async transfer per batch into a pinned sink,
and the buffers are double buffered so host packing and blending overlap with the
previous batch's compute. Edge-tile padding moved to the host, which is
bit-identical to padding on the device and keeps the transfer contiguous.

Utilization is now a steady 100%, worth about 1.08x end to end (1.286s to 1.180s
on a 96x384x384 volume, median of six runs after warmup). Blending order is
unchanged, so `batch_size=1` remains bit-identical; verified on CPU, where no
cuDNN kernel selection is involved, that the pipelined and serial loops agree
exactly at batch sizes 1 and 4.

### Performance notes

Profiling the tile loop with AMP enabled, on a 64x256x256 volume at a 32x64x64
tile: model forward 83.5%, host-to-device 4.4%, device-to-host 3.2%, host
accumulation 6.9%, final divide 1.8%. That bounded the pipelining work above at
roughly 1.1x, which is what it delivered. An earlier note in this file inferred
that transfer dominated for large tiles; that was wrong. Large tiles gain less
from batching because they already saturate the GPU, not because of transfer.

The remaining runtime is the model forward itself. Beyond mixed precision, going
faster would mean a smaller architecture, channels-last layouts, or an exported
engine, none of which are in scope here.

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
