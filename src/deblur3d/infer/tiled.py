# src/deblur3d/infer/tiled.py
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["deblur_volume_tiled", "validate_tiling", "validate_volume_shape"]

MIN_VOLUME_SIZE = 16


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

# Throughput saturates by 8..16 tiles per forward; past that the batch only costs
# memory. Measured 1.49x at 32x64x64, 1.12x at 64x128x128.
_MAX_BATCH = 16

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
    return max(1, min(_MAX_BATCH, int((free * _VRAM_SAFETY) // per_tile)))


def _starts(L: int, tile: int, overlap: int) -> list[int]:
    # step for interior tiles
    step = tile - overlap if tile < L else L
    # initial evenly spaced starts
    starts = list(range(0, max(1, L - tile + 1), step))
    # ensure the last tile reaches the border
    last = max(0, L - tile)
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def deblur_volume_tiled(
    net: torch.nn.Module,
    vol: Union[np.ndarray, torch.Tensor],
    tile: Tuple[int, int, int] = (96, 128, 128),
    overlap: Tuple[int, int, int] = (24, 32, 32),
    device: str = "cuda",
    use_amp: bool = False,              # PT1.12 + InstanceNorm: keep False unless you use GroupNorm
    pad_mode: str = "reflect",
    clamp01: bool = True,
    batch_size: Union[int, str] = "auto",
) -> np.ndarray:
    """
    Tiled 3D inference with Hann blending.

    Args:
        net:   3D residual model expecting (N,1,D,H,W) in [0,1].
        vol:   (D,H,W) float32 numpy or torch tensor in [0,1].
        tile:  (Dz, Dy, Dx) tile size.
        overlap: (Oz, Oy, Ox) overlap for blending.
        device: "cuda" or "cpu".
        use_amp: enable CUDA autocast (set False for InstanceNorm on PT1.12).
        pad_mode: pad mode for edge tiles ("reflect" | "replicate" | "constant").
        clamp01: clamp output to [0,1] before returning.
        batch_size: tiles per forward pass. "auto" sizes it from free VRAM; an
            int forces a value. Batching does not change the tiling grid, so it
            leaves blending untouched, but cuDNN may select a different kernel
            per batch size, which perturbs results at ~1e-5.

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

    zs = _starts(D, td, od)
    ys = _starts(H, th, oh)
    xs = _starts(W, tw, ow)
    coords = [(z, y, x) for z in zs for y in ys for x in xs]

    if isinstance(batch_size, str):
        if batch_size != "auto":
            raise ValueError(f"batch_size must be an int or 'auto'; got {batch_size!r}.")
        bs = _auto_batch_size(net, device_t, td * th * tw, minimum_tile_size)
    else:
        bs = max(1, int(batch_size))

    def _run(chunk):
        patches = []
        for z, y, x in chunk:
            p = v[:, :, z:z+td, y:y+th, x:x+tw].to(device_t, non_blocking=True)
            if p.shape[2:] != (td, th, tw):
                padz = td - p.shape[2]
                pady = th - p.shape[3]
                padx = tw - p.shape[4]
                p = F.pad(p, (0, padx, 0, pady, 0, padz), mode=pad_mode)
            patches.append(p)

        with autocast(enabled=(use_amp and device_t.type == "cuda")):
            preds = net(torch.cat(patches, dim=0) if len(patches) > 1 else patches[0])

        # Accumulate in coordinate order, matching the single-tile path exactly.
        for j, (z, y, x) in enumerate(chunk):
            pd = min(td, D - z); ph = min(th, H - y); pw = min(tw, W - x)
            weighted = (preds[j:j+1, :, :pd, :ph, :pw] * w3[:, :, :pd, :ph, :pw]).to("cpu")
            out[:, :, z:z+pd, y:y+ph, x:x+pw] += weighted
            wei[:, :, z:z+pd, y:y+ph, x:x+pw] += w3_cpu[:, :, :pd, :ph, :pw]

    i = 0
    while i < len(coords):
        chunk = coords[i:i + bs]
        try:
            _run(chunk)
        except RuntimeError as e:
            # The budget is an estimate; back off rather than failing the run.
            if bs > 1 and "out of memory" in str(e).lower():
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            raise
        i += len(chunk)

    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    res = (out / wei.clamp_min(torch.finfo(wei.dtype).tiny)).squeeze(0).squeeze(0)
    if clamp01:
        res = res.clamp(0, 1)
    return res.detach().numpy().astype(np.float32)
