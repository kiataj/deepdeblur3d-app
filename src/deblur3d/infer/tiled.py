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

    Returns:
        (D,H,W) float32 numpy array.
    """
    net.eval()
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # normalize input → torch on device
    if isinstance(vol, np.ndarray):
        v = torch.from_numpy(vol)
    else:
        v = vol

    down_layers = getattr(net, "down", None)
    minimum_tile_size = 2 ** len(down_layers) if down_layers is not None else MIN_VOLUME_SIZE
    validate_volume_shape(v.shape, minimum_size=minimum_tile_size)
    tile, overlap = validate_tiling(tile, overlap, minimum_size=minimum_tile_size)
    v = v.to(device_t, dtype=torch.float32, non_blocking=True).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    D, H, W = int(v.shape[2]), int(v.shape[3]), int(v.shape[4])
    td, th, tw = (min(size, extent) for size, extent in zip(tile, (D, H, W)))
    od, oh, ow = map(int, overlap)

    out = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=device_t)
    wei = torch.zeros_like(out)

    def _hann(sz: int, ov: int):
        if sz <= 1 or ov <= 0:
            return torch.ones(sz, device=device_t)
        g = torch.hann_window(sz, periodic=False, device=device_t)
        return g.clamp_min(1e-6)

    wz, wy, wx = _hann(td, od), _hann(th, oh), _hann(tw, ow)
    w3 = wz.view(1, 1, td, 1, 1) * wy.view(1, 1, 1, th, 1) * wx.view(1, 1, 1, 1, tw)

    step_z = td - od if td < D else D
    step_y = th - oh if th < H else H
    step_x = tw - ow if tw < W else W

    # Import here to avoid requiring CUDA on CPUs
    from torch.cuda.amp import autocast

    zs = _starts(D, td, od)
    ys = _starts(H, th, oh)
    xs = _starts(W, tw, ow)

    for z in zs:
        for y in ys:
            for x in xs:
                patch = v[:, :, z:z+td, y:y+th, x:x+tw]
                if patch.shape[2:] != (td, th, tw):
                    padz = td - patch.shape[2]
                    pady = th - patch.shape[3]
                    padx = tw - patch.shape[4]
                    patch = F.pad(patch, (0, padx, 0, pady, 0, padz), mode=pad_mode)

                with autocast(enabled=(use_amp and device_t.type == "cuda")):
                    pred = net(patch)

                pd = min(td, D - z); ph = min(th, H - y); pw = min(tw, W - x)
                w  = w3[:, :, :pd, :ph, :pw]
                pred = pred[:, :, :pd, :ph, :pw]

                out[:, :, z:z+pd, y:y+ph, x:x+pw] += pred * w
                wei[:, :, z:z+pd, y:y+ph, x:x+pw] += w


    res = (out / wei.clamp_min(torch.finfo(wei.dtype).tiny)).squeeze(0).squeeze(0)
    if clamp01:
        res = res.clamp(0, 1)
    return res.detach().cpu().numpy().astype(np.float32)
