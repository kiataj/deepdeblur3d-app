# Changelog

This project follows [Semantic Versioning](https://semver.org/).

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

- **Mixed precision is on by default.**
- **The GUI is down from 16 controls to 6.**

## v2.0.0

Tagged during development and superseded by v3.0.0, which includes all of it.
