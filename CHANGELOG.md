# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## v3.0.0 (2026-07-31)

Major, because default output changes. To reproduce earlier results, pass
`--no-amp` and `--legacy-borders` and keep the "Balanced (default)" preset.

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
- The CPU fallback no longer retries while the failed CUDA allocation is held,
  and aborting no longer prints a traceback.

### Added

- **Command line interface.** `deblur3d IN OUT`, or `--batch` over a folder.
  Accepts TIFF stacks, `.npy`, a directory of slices, and a directory of such
  directories. Existing outputs are skipped unless `--overwrite` is given.
- **Progress reporting and an abort button**, in both front ends.
- **Automatic batch sizing** from free VRAM, measured with a probe forward rather
  than assumed, backing off if a batch still will not fit, and scaling with the
  GPU's SM count.
- **Update notification** in both front ends, showing the release notes and the
  commands to update. It does not install anything.
- Version derived from git tags, and provenance recorded on every result: app
  version, model revision, preset and parameters.
- MIT licence.

### Changed

- **Mixed precision is on by default, worth 1.87x.** It was off behind a comment
  reading "keep False unless you use GroupNorm"; the model is GroupNorm
  throughout, so the case named as safe was the one that held. Costs about 6e-4.
- **The GUI is down from 16 controls to 6**, with tiling presets replacing six
  spin boxes and an info badge describing them.
- The "Low memory" preset's Z overlap is doubled to 16. A quarter of the tile
  depth was too thin to blend away tile-edge error.
- The tile loop is pipelined through pinned buffers, taking GPU utilisation from
  74-100% to a steady 100%.
- Build artifacts and per-machine settings are no longer tracked, so releases
  version cleanly.

### Deliberately not done

- **Automatic tile sizing from VRAM.** Tile size changes the result, so deriving
  it from the GPU would make output depend on the machine.
- **Threading the host work.** Torch already threads it, and at per-tile sizes
  that costs 1.7x rather than saving anything.

### Known limitations

- `torch==1.12.1+cu116` builds for sm_37 to sm_86, so 40-series and 50-series
  GPUs cannot run this until the pin is raised.
- The default preset's 50% overlap in Y and X is the slowest option; "Low memory"
  is both the lightest and the quickest.

## v2.0.0

Tagged during development and superseded by v3.0.0, which includes all of it.
