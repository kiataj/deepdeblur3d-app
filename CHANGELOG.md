# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## v3.0.3 (2026-08-10)

### Added

- **A one-click update.** The update notice now has an "Update now" button that
  installs the release here and streams the commands' output to a dialog, then
  asks for a restart. It refuses while an inference is running, and refuses on a
  checkout with uncommitted changes rather than overwriting them.

### Fixed

- The update dialog interpolated the whole release payload into one line, since
  it still treated it as a string after the check began returning a record. It
  now shows the title, the notes behind "Show Details", and the commands.
- The fallback update command pointed at PyPI, where this package is not
  published, so it failed for anyone who had not cloned. It now installs from
  the tag, which needs no clone.

## v3.0.2 (2026-08-09)

### Fixed

- The control parameter fields rendered blank on some installs, their value
  readable only once selected. The readout styled its background with
  `palette(base)`, which resolves against the Qt palette and so ignores the
  napari theme, while the text colour came from napari's own stylesheet. Where
  the two happened to coincide the text was invisible. The style now sets only
  the frame and leaves both colours to the theme.

## v3.0.1 (2026-08-01)

### Fixed

- The spin arrows on the control readouts moved the displayed number without
  changing the value used for inference, so a nudged strength ran at the old one.
- numpy 2 promoted normalized float volumes to float64, doubling their memory in
  the same path the out-of-memory work targeted. Integer inputs were unaffected,
  which is why it did not show up in a TIFF run.

### Added

- numpy 2 is supported; the `numpy<2` pin is gone.
- CI: the suite runs on Python 3.10 and 3.12 across torch 1.12.1 and 2.9.1 and
  numpy 1 and 2, plus a wheel build that rejects a version built from a dirty tree.

## v3.0.0 (2026-07-31)

Major, because default output changes. To reproduce earlier results, pass
`--legacy-borders` and keep the "Balanced (default)" preset.

### Fixed

- **Out of memory on large volumes.** GPU memory scaled with the volume, not the
  tile, so tiling did not bound it. The volume and blending accumulators now stay
  in host memory and the controls are applied in Z-slabs. Peak VRAM is flat at
  0.41 GB from 4.2M to 226.5M voxels, against 0.46 to 3.13 GB before.
- **The volume border was an unblended tile-edge prediction.** The outermost
  slice, row and column of each axis were covered by a single tile at a blending
  weight of 1e-6, against about 0.98 inside, so they were raw predictions from a
  tile's own edge. The grid now nudges its outermost tiles past each end and
  reflects the part outside. On real micro-CT data the last slice went from 4.66x
  the interior high-frequency energy to 1.08x.
- **The residual cache ignored the tiling settings**, so switching preset reused
  the residual from the previous tiling grid. The key now covers tile, overlap,
  pad mode, clamping, AMP, batch size and border margin.
- **The residual cache pinned VRAM for the whole session.** Entries are now
  host-resident and capped at one.
- The CPU fallback no longer retries while the failed CUDA allocation is held.

### Added

- **Command line interface.**
- **Progress reporting and an abort button.**
- **Automatic batch sizing** from free VRAM, measured with a probe forward rather
  than assumed, backing off if a batch still will not fit, and scaling with the
  GPU's SM count.
- **Update notification** in both front ends, showing the release notes and the
  commands to update. It does not install anything.

### Changed

- **Mixed precision is off by default**, opt in with `--amp`. It is faster on
  torch 1.12 but slower on torch 2.x, where fp32 beats it.
- **The GUI is down from 16 controls to 6.**

## v2.0.0

Tagged during development and superseded by v3.0.0, which includes all of it.
