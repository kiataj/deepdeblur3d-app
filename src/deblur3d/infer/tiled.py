# src/deblur3d/infer/tiled.py
from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "deblur_volume_tiled",
    "validate_tiling",
    "validate_volume_shape",
    "InferenceAborted",
    "amp_is_safe",
]

MIN_VOLUME_SIZE = 16


class InferenceAborted(Exception):
    """Raised when a caller's abort callback asks inference to stop.

    Deliberately not a RuntimeError: the tile loop retries RuntimeErrors that
    look like OOM, and an abort must never be mistaken for one.
    """


def amp_is_safe(net: torch.nn.Module) -> bool:
    """Whether autocast can be enabled for this model.

    Half precision is unreliable for InstanceNorm on PyTorch 1.12, which is why
    AMP used to be off by default. This model is GroupNorm throughout, so the
    exception does not apply; check rather than assume, since the checkpoint's
    architecture is read from config.json and could change.
    """
    return not any(
        isinstance(m, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d))
        for m in net.modules()
    )


def validate_volume_shape(shape, minimum_size: int = MIN_VOLUME_SIZE) -> tuple[int, int, int]:
    """Validate a volume against the four-level 3D U-Net input requirements."""
    shape = tuple(int(size) for size in shape)
    if len(shape) != 3:
        raise ValueError(
            f"DeepDeblur3D requires a 3D volume with shape (Z, Y, X); got {shape}. "
            "Load TIFF slices as one stack (Napari: Open Files as Stack)."
        )

    axes = ("Z", "Y", "X")
    too_small = [f"{axis}={size}" for axis, size in zip(axes, shape) if size < minimum_size]
    if too_small:
        raise ValueError(
            f"DeepDeblur3D requires at least {minimum_size} voxels in every dimension "
            f"(Z, Y, X) because of the model's downsampling stages; got {shape} "
            f"({', '.join(too_small)} is too small). Load at least {minimum_size} slices "
            "as one 3D stack when Z is the limiting dimension."
        )
    return shape


def validate_tiling(
    tile, overlap, minimum_size: int = MIN_VOLUME_SIZE
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Validate tile sizes and overlaps before allocating inference tensors."""
    tile = tuple(int(size) for size in tile)
    overlap = tuple(int(size) for size in overlap)
    if len(tile) != 3 or len(overlap) != 3:
        raise ValueError(
            f"Tile and overlap must both have shape (Z, Y, X); got "
            f"tile={tile}, overlap={overlap}."
        )

    axes = ("Z", "Y", "X")
    too_small = [f"{axis}={size}" for axis, size in zip(axes, tile) if size < minimum_size]
    if too_small:
        raise ValueError(
            f"Each tile dimension must be at least {minimum_size} voxels for this model; "
            f"got tile={tile} ({', '.join(too_small)} is too small)."
        )

    invalid_overlaps = [
        f"{axis}: overlap={ov}, tile={size}"
        for axis, ov, size in zip(axes, overlap, tile)
        if ov < 0 or ov >= size
    ]
    if invalid_overlaps:
        raise ValueError(
            "Each overlap must be non-negative and smaller than its tile dimension; "
            + "; ".join(invalid_overlaps)
            + "."
        )
    return tile, overlap


# Activation memory is linear in tile voxels for a fixed architecture, so one
# probe forward gives the slope. Measured once per (model, device) per session;
# the probe costs ~10 ms because the CUDA context and cuDNN kernels it loads are
# one-time costs the first real tile would otherwise pay.
_BYTES_PER_VOXEL: dict = {}

# Fraction of free VRAM the batch is allowed to occupy. Extrapolating from the
# probe under-predicts by up to ~12% on tile shapes that are not powers of two,
# and the caching allocator fragments, so this headroom is load-bearing.
_VRAM_SAFETY = 0.8

# Throughput saturates once there is enough parallel work to fill the SMs; past
# that the batch only costs memory. Measured 1.49x at 32x64x64 and 1.12x at
# 64x128x128 on a 38-SM card, saturating by 8..16 tiles, hence ~0.4 tiles per SM.
#
# The slope is extrapolated from that one GPU, so the floor keeps every card at
# least at the value it was measured with: a larger GPU may get a larger batch,
# none gets a smaller one. Passing an explicit batch_size bypasses this entirely.
_MAX_BATCH = 16
_MAX_BATCH_PER_SM = 0.4


def _batch_cap(device_t: torch.device) -> int:
    """Upper bound on tiles per forward, scaled to the GPU's SM count."""
    try:
        index = device_t.index if device_t.index is not None else torch.cuda.current_device()
        sm_count = torch.cuda.get_device_properties(index).multi_processor_count
    except (RuntimeError, AttributeError, AssertionError):
        return _MAX_BATCH
    return max(_MAX_BATCH, int(sm_count * _MAX_BATCH_PER_SM))

# Small probes are dominated by fixed overheads and read high, so keep it at
# least this large in each axis.
_PROBE_TILE = (32, 64, 64)


@torch.no_grad()
def _bytes_per_voxel(net: torch.nn.Module, device_t: torch.device, minimum_size: int) -> float:
    """Measure activation bytes per tile voxel with one probe forward."""
    key = (id(net), str(device_t))
    cached = _BYTES_PER_VOXEL.get(key)
    if cached is not None:
        return cached

    probe = tuple(max(size, minimum_size) for size in _PROBE_TILE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_t)
    # Baseline excludes the model weights, which are a fixed cost and would
    # otherwise be smeared across the probe's voxels and inflate the slope.
    baseline = torch.cuda.memory_allocated(device_t)
    x = torch.zeros((1, 1, *probe), device=device_t)
    net(x)
    activation = torch.cuda.max_memory_allocated(device_t) - baseline
    del x
    torch.cuda.empty_cache()

    bpv = activation / float(probe[0] * probe[1] * probe[2])
    _BYTES_PER_VOXEL[key] = bpv
    return bpv


def _auto_batch_size(
    net: torch.nn.Module, device_t: torch.device, tile_voxels: int, minimum_size: int
) -> int:
    """Largest number of tiles per forward that fits the free VRAM budget."""
    if device_t.type != "cuda":
        return 1
    # mem_get_info wants a device index; a bare torch.device("cuda") is rejected.
    index = device_t.index if device_t.index is not None else torch.cuda.current_device()
    try:
        bpv = _bytes_per_voxel(net, device_t, minimum_size)
        torch.cuda.empty_cache()
        free, _total = torch.cuda.mem_get_info(index)
    except (RuntimeError, AttributeError):
        # mem_get_info is absent on some builds; a single tile always fit before.
        return 1
    per_tile = max(1.0, bpv * tile_voxels)
    return max(1, min(_batch_cap(device_t), int((free * _VRAM_SAFETY) // per_tile)))


def _starts(L: int, tile: int, overlap: int, margin: int = 0) -> list[int]:
    """Tile origins along one axis.

    With `margin > 0` the outermost tiles are nudged past the ends of the volume
    so the outermost real voxels sit inside a tile rather than on its edge. The
    interior grid is untouched and no tiles are added, so this costs nothing; it
    stays gap-free because the margin never exceeds the overlap. Starts may be
    negative or run past L, and the caller reflects the out-of-volume part.
    """
    # step for interior tiles
    step = tile - overlap if tile < L else L
    # initial evenly spaced starts
    starts = list(range(0, max(1, L - tile + 1), step))
    # ensure the last tile reaches the border
    last = max(0, L - tile)
    if starts[-1] != last:
        starts.append(last)

    if margin <= 0:
        return starts

    def shifted(m: int) -> list[int]:
        if len(starts) == 1:
            # A single tile spans the axis, so one shifted copy cannot keep both
            # ends off an edge. This is the only case that costs an extra tile.
            return [-m, m]
        return [starts[0] - m] + starts[1:-1] + [starts[-1] + m]

    def covers(candidate: list[int]) -> bool:
        if candidate[0] > 0:
            return False
        reach = candidate[0] + tile
        for s in candidate[1:]:
            # `s == reach` is a gapless but unblended seam, which looks worse
            # than the border artifact this shift exists to remove.
            if s >= reach:
                return False
            reach = max(reach, s + tile)
        return reach >= L

    # Pushing both ends outward can pull a sparse grid apart, so take the largest
    # shift that still covers every voxel rather than assuming the full one fits.
    for m in range(margin, 0, -1):
        candidate = shifted(m)
        if covers(candidate):
            return candidate
    return starts


def _border_margin(tile: Tuple[int, int, int], overlap: Tuple[int, int, int],
                   minimum_size: int) -> tuple[int, int, int]:
    """How far to extend the grid past each end of the volume.

    A voxel on a tile's outer face is produced from zero-padded convolutions and
    is unreliable. Interior tiles hide that behind an overlapping neighbour, but
    the volume's own border has no neighbour, so its outermost slice was the raw
    edge prediction at a blending weight of 1e-6 against ~0.98 inside.

    One coarsest-level voxel (2**levels) is the scale over which that padding
    contaminates the result.

    Capped at half the overlap: the outermost tiles are shifted outward, which
    spends overlap at the seam behind them. Spending all of it leaves those two
    tiles merely abutting, replacing the border artifact with a worse unblended
    seam (measured 49% against 4% at half).
    """
    return tuple(
        0 if ov < 2 else max(1, min(minimum_size, t // 4, ov // 2))
        for t, ov in zip(tile, overlap)
    )


@torch.no_grad()
def deblur_volume_tiled(
    net: torch.nn.Module,
    vol: Union[np.ndarray, torch.Tensor],
    tile: Tuple[int, int, int] = (96, 128, 128),
    overlap: Tuple[int, int, int] = (24, 32, 32),
    device: str = "cuda",
    use_amp: Union[bool, str] = "auto",
    pad_mode: str = "reflect",
    clamp01: bool = True,
    batch_size: Union[int, str] = "auto",
    border_margin: Union[int, str, Tuple[int, int, int]] = "auto",
    progress: Optional[Callable[[int, int], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """
    Tiled 3D inference with Hann blending.

    Args:
        net:   3D residual model expecting (N,1,D,H,W) in [0,1].
        vol:   (D,H,W) float32 numpy or torch tensor in [0,1].
        tile:  (Dz, Dy, Dx) tile size.
        overlap: (Oz, Oy, Ox) overlap for blending.
        device: "cuda" or "cpu".
        use_amp: enable CUDA autocast. "auto" turns it on whenever the device is
            CUDA and the model has no InstanceNorm. Roughly 1.8x faster, at a
            numerical cost around 6e-4.
        pad_mode: pad mode for edge tiles ("reflect" | "replicate" | "constant").
        clamp01: clamp output to [0,1] before returning.
        batch_size: tiles per forward pass. "auto" sizes it from free VRAM; an
            int forces a value. Batching does not change the tiling grid, so it
            leaves blending untouched, but cuDNN may select a different kernel
            per batch size, which perturbs results at ~1e-5.
        border_margin: how far to extend the tile grid past each end of the
            volume so its outermost voxels are not produced from a tile's own
            edge. "auto" derives it from the model depth and the overlap; 0
            restores the pre-2.1 behaviour, where the first and last slice of
            each axis were unblended tile-edge predictions.
        progress: called as progress(tiles_done, tiles_total) after each batch.
        should_abort: polled before each batch; if it returns True the run stops
            and raises InferenceAborted.

    Returns:
        (D,H,W) float32 numpy array.

    The volume and the blending accumulators stay in host memory; only the current
    batch of tiles is resident on `device`, so device memory is O(batch * tile),
    not O(volume).
    """
    net.eval()
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # normalize input → torch on host
    if isinstance(vol, np.ndarray):
        v = torch.from_numpy(vol)
    else:
        v = vol

    down_layers = getattr(net, "down", None)
    minimum_tile_size = 2 ** len(down_layers) if down_layers is not None else MIN_VOLUME_SIZE
    validate_volume_shape(v.shape, minimum_size=minimum_tile_size)
    tile, overlap = validate_tiling(tile, overlap, minimum_size=minimum_tile_size)
    v = v.to("cpu", dtype=torch.float32).reshape(1, 1, *v.shape)  # (1,1,D,H,W)

    D, H, W = int(v.shape[2]), int(v.shape[3]), int(v.shape[4])
    td, th, tw = (min(size, extent) for size, extent in zip(tile, (D, H, W)))
    od, oh, ow = map(int, overlap)

    out = torch.zeros((1, 1, D, H, W), dtype=torch.float32)
    wei = torch.zeros_like(out)

    def _hann(sz: int, ov: int):
        if sz <= 1 or ov <= 0:
            return torch.ones(sz, device=device_t)
        g = torch.hann_window(sz, periodic=False, device=device_t)
        return g.clamp_min(1e-6)

    wz, wy, wx = _hann(td, od), _hann(th, oh), _hann(tw, ow)
    w3 = wz.view(1, 1, td, 1, 1) * wy.view(1, 1, 1, th, 1) * wx.view(1, 1, 1, 1, tw)
    w3_cpu = w3.to("cpu")

    step_z = td - od if td < D else D
    step_y = th - oh if th < H else H
    step_x = tw - ow if tw < W else W

    # Import here to avoid requiring CUDA on CPUs
    from torch.cuda.amp import autocast

    if isinstance(border_margin, str):
        if border_margin != "auto":
            raise ValueError(
                f"border_margin must be an int, a 3-tuple, or 'auto'; got {border_margin!r}."
            )
        mz, my, mx = _border_margin((td, th, tw), (od, oh, ow), minimum_tile_size)
    elif isinstance(border_margin, int):
        mz = my = mx = max(0, border_margin)
    else:
        mz, my, mx = (max(0, int(m)) for m in border_margin)

    def _clip(start: int, size: int, extent: int):
        """Map a tile that may hang off either end onto the volume."""
        d0, d1 = max(0, start), min(extent, start + size)
        return d0, d1, d0 - start, d0 - start + (d1 - d0)

    zs = _starts(D, td, od, mz)
    ys = _starts(H, th, oh, my)
    xs = _starts(W, tw, ow, mx)
    coords = [(z, y, x) for z in zs for y in ys for x in xs]

    if isinstance(batch_size, str):
        if batch_size != "auto":
            raise ValueError(f"batch_size must be an int or 'auto'; got {batch_size!r}.")
        bs = _auto_batch_size(net, device_t, td * th * tw, minimum_tile_size)
    else:
        bs = max(1, int(batch_size))

    if isinstance(use_amp, str):
        if use_amp != "auto":
            raise ValueError(f"use_amp must be a bool or 'auto'; got {use_amp!r}.")
        amp_enabled = device_t.type == "cuda" and amp_is_safe(net)
    else:
        amp_enabled = bool(use_amp) and device_t.type == "cuda"

    # Two-stage pipeline. Pageable host memory forces every copy to synchronize,
    # and a per-tile copy back costs one sync per tile, so the GPU sat idle
    # between batches (measured 74-100% SM, mean ~90%). Staging through pinned
    # buffers lets the copies run async on their own stream, and double buffering
    # overlaps host packing and accumulation with the previous batch's compute.
    use_cuda = device_t.type == "cuda"
    copy_stream = torch.cuda.Stream() if use_cuda else None

    def _alloc(n: int):
        return torch.empty((n, 1, td, th, tw), dtype=torch.float32, pin_memory=use_cuda)

    def _pack(stage: torch.Tensor, chunk) -> torch.Tensor:
        """Pack tiles into one contiguous host batch, padding edge tiles.

        Padding on the host is bit-identical to padding on the device and keeps
        the transfer a single contiguous copy.
        """
        for j, (z, y, x) in enumerate(chunk):
            (dz0, dz1, lz0, lz1) = _clip(z, td, D)
            (dy0, dy1, ly0, ly1) = _clip(y, th, H)
            (dx0, dx1, lx0, lx1) = _clip(x, tw, W)
            p = v[:, :, dz0:dz1, dy0:dy1, dx0:dx1]
            pad = (lx0, tw - lx1, ly0, th - ly1, lz0, td - lz1)
            if any(pad):
                p = F.pad(p, pad, mode=pad_mode)
            stage[j:j+1].copy_(p)
        return stage[:len(chunk)]

    def _accumulate(chunk, host_out: torch.Tensor):
        """Blend a completed batch, in coordinate order, matching batch=1.

        Tiles may hang off either end of the volume; only the in-range part is
        written back, so the reflected margin contributes nothing to the result.
        """
        for j, (z, y, x) in enumerate(chunk):
            (dz0, dz1, lz0, lz1) = _clip(z, td, D)
            (dy0, dy1, ly0, ly1) = _clip(y, th, H)
            (dx0, dx1, lx0, lx1) = _clip(x, tw, W)
            out[:, :, dz0:dz1, dy0:dy1, dx0:dx1] += host_out[
                j:j+1, :, lz0:lz1, ly0:ly1, lx0:lx1
            ]
            wei[:, :, dz0:dz1, dy0:dy1, dx0:dx1] += w3_cpu[
                :, :, lz0:lz1, ly0:ly1, lx0:lx1
            ]

    def _submit(stage: torch.Tensor, sink: torch.Tensor, chunk):
        """Run one batch, leaving the copy back in flight."""
        host_in = _pack(stage, chunk)
        if use_cuda:
            with torch.cuda.stream(copy_stream):
                gpu_in = host_in.to(device_t, non_blocking=True)
            compute = torch.cuda.current_stream()
            compute.wait_stream(copy_stream)
            # Allocated on copy_stream but consumed on the compute stream; without
            # this the allocator may hand the block out again before the forward
            # has read it.
            gpu_in.record_stream(compute)
        else:
            gpu_in = host_in

        with autocast(enabled=amp_enabled):
            preds = net(gpu_in)
        # Autocast returns half; blending and accumulation stay in float32.
        weighted = preds.float() * w3

        sink[:len(chunk)].copy_(weighted, non_blocking=use_cuda)
        if not use_cuda:
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    total = len(coords)
    if progress is not None:
        progress(0, total)

    stages = [_alloc(bs), _alloc(bs)]
    sinks = [_alloc(bs), _alloc(bs)]
    events: list = [None, None]
    pending: list = [None, None]
    slot = 0
    done = 0

    def _retire(s: int) -> int:
        """Wait for slot `s`'s copy back, then blend it. Returns tiles retired."""
        if pending[s] is None:
            return 0
        if events[s] is not None:
            events[s].synchronize()
        chunk = pending[s]
        _accumulate(chunk, sinks[s])
        pending[s] = None
        events[s] = None
        return len(chunk)

    i = 0
    while i < total:
        if should_abort is not None and should_abort():
            raise InferenceAborted(f"Inference aborted after {done} of {total} tiles.")
        chunk = coords[i:i + bs]
        try:
            events[slot] = _submit(stages[slot], sinks[slot], chunk)
        except RuntimeError as e:
            # The budget is an estimate; back off rather than failing the run.
            if bs > 1 and "out of memory" in str(e).lower():
                done += _retire(1 - slot)
                pending[slot] = None
                events[slot] = None
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                stages = [_alloc(bs), _alloc(bs)]
                sinks = [_alloc(bs), _alloc(bs)]
                slot = 0
                continue
            raise
        pending[slot] = chunk
        i += len(chunk)

        # Blend the previous batch while this one is still on the GPU.
        done += _retire(1 - slot)
        if progress is not None:
            progress(done, total)
        slot = 1 - slot

    for s in (1 - slot, slot):
        done += _retire(s)
    if progress is not None:
        progress(done, total)

    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    res = (out / wei.clamp_min(torch.finfo(wei.dtype).tiny)).squeeze(0).squeeze(0)
    if clamp01:
        res = res.clamp(0, 1)
    return res.detach().numpy().astype(np.float32)
