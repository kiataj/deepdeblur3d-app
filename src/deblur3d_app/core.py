"""Headless inference core: model resolution, I/O, and the control pipeline.

Deliberately free of napari and Qt imports so the CLI can run without a display.
Anything that needs to ask the user something takes a callback.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tifffile as tiff
import torch
from packaging.version import InvalidVersion, Version
from platformdirs import user_data_dir
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import HfApi, hf_hub_download

try:
    from huggingface_hub.errors import EntryNotFoundError
except Exception:
    try:
        from huggingface_hub import EntryNotFoundError  # type: ignore
    except Exception:
        class EntryNotFoundError(Exception): ...

from deblur3d.data.io import read_volume_float01
from deblur3d.infer.tiled import (
    InferenceAborted,
    deblur_volume_tiled,
    validate_volume_shape,
)
from deblur3d.models import ControlledUNet3D, UNet3D_Residual

from . import __version__

HF_REPO_ID = os.getenv("DEBLUR3D_HF_REPO", "HippoCanFly/DeepDeBlur3D")
HF_FILENAME = os.getenv("DEBLUR3D_HF_FILE", "pytorch_model.safetensors")
HF_REVISION_DEFAULT = os.getenv("DEBLUR3D_HF_REV", "v1.0.0")
GITHUB_REPO = os.getenv("DEBLUR3D_GITHUB_REPO", "kiataj/deepdeblur3d-app")

APP_AUTHOR = "DeepDeBlur3D"
APP_NAME = "deblur3d-gui"
STATE_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "model_state.json"

# Named tile/overlap pairs. The GUI offers these instead of six spin boxes.
# "Balanced" reproduces the historical default exactly; changing tile or overlap
# changes the tiling grid and therefore the result, so presets are fixed and
# recorded in provenance rather than derived from the machine.
#
# Z overlap is held at half the tile depth. A quarter is not enough to blend away
# the tile-edge error the model produces from its zero-padded convolutions:
# measured 16% slice-to-slice deviation at overlap 8 on a 32-deep tile, against
# 4.7% at 16.
TILE_PRESETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "Balanced (default)": ((64, 256, 256), (32, 128, 128)),
    "Low memory":         ((32, 128, 128), (16, 32, 32)),
    "Fast (less overlap)": ((64, 256, 256), (16, 64, 64)),
}
DEFAULT_PRESET = "Balanced (default)"

# Per-tile VRAM is ~708 bytes/voxel for this model. Neither runtimes nor blending
# scores are quoted: the first depends on the GPU and the second is a synthetic
# measurement that would read as a quality guarantee. Relative wording is what
# actually helps someone choose.
TILE_PRESET_HELP: dict[str, str] = {
    "Balanced (default)": (
        "Tile 64x256x256, overlap 32x128x128, about 3.0 GB of VRAM per tile.\n"
        "The widest overlap, giving the smoothest blending, and the slowest, "
        "since the model revisits each voxel several times.\n"
        "The only preset that reproduces output from before v3.0.\n"
        "Use for final and published results, and to match earlier runs."
    ),
    "Low memory": (
        "Tile 32x128x128, overlap 16x32x32, about 0.4 GB of VRAM per tile.\n"
        "Eight times less VRAM than the others, and the quickest, because small "
        "tiles with a narrow overlap add up to less work. Blending is a little "
        "looser than Balanced.\n"
        "Use on GPUs with limited memory, or for very large volumes."
    ),
    "Fast (less overlap)": (
        "Tile 64x256x256, overlap 16x64x64, about 3.0 GB of VRAM per tile.\n"
        "The same large tiles as Balanced but a narrower overlap, so there are "
        "fewer tiles to get through. Blending is a little looser than Balanced.\n"
        "Use when you want Balanced's large tiles in less time."
    ),
}

Prompt = Callable[[str, str], bool]


# ----------------------------- HuggingFace -----------------------------
@dataclass
class HFModelSpec:
    repo_id: str
    weights_filename: str
    config_filename: str = "config.json"
    revision: Optional[str] = None


def has_internet(timeout: float = 2.0, host: str = "huggingface.co") -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout)
        return True
    except OSError:
        return False


def load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(d: dict):
    try:
        STATE_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def parse_semver_tag(tag: str) -> Optional[Version]:
    s = tag[1:] if tag.startswith("v") else tag
    try:
        v = Version(s)
        return v if not v.is_prerelease else None
    except InvalidVersion:
        return None


def _latest_semver_tag(api: HfApi, repo_id: str) -> Optional[str]:
    refs = api.list_repo_refs(repo_id)
    best: Optional[tuple[Version, str]] = None
    for t in refs.tags:
        v = parse_semver_tag(t.name)
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, t.name)
    return best[1] if best else None


def _desired_revision(api: HfApi, repo_id: str, prompt: Optional[Prompt]) -> str:
    state = load_state()
    current = state.get("revision") or HF_REVISION_DEFAULT
    cur_ver = parse_semver_tag(current)
    try:
        latest = _latest_semver_tag(api, repo_id)
    except Exception:
        latest = None
    if latest and prompt is not None:
        lat_ver = parse_semver_tag(latest)
        if cur_ver and lat_ver and lat_ver > cur_ver:
            if prompt(
                "Model update available",
                f"A newer tagged model ({latest}) is available.\n\n"
                f"Update from {current} to {latest}?",
            ):
                current = latest
    if state.get("revision") != current:
        state["revision"] = current
        save_state(state)
    return current


def ensure_model_assets(
    spec: HFModelSpec, prompt: Optional[Prompt] = None
) -> Tuple[str, Optional[str]]:
    """Resolve weights and config, downloading them if needed.

    `prompt` is asked before moving to a newer model revision. Without one (the
    CLI), the pinned revision is kept rather than silently upgrading, so a batch
    run cannot change models halfway through a study.
    """
    api = HfApi()
    if has_internet():
        desired_rev = _desired_revision(api, spec.repo_id, prompt)
    else:
        desired_rev = load_state().get("revision") or HF_REVISION_DEFAULT

    st = load_state()
    last_rev = st.get("revision")
    force = (last_rev is not None) and (desired_rev != last_rev)

    weights_path = hf_hub_download(
        spec.repo_id, spec.weights_filename, revision=desired_rev, force_download=force
    )
    try:
        config_path = hf_hub_download(
            spec.repo_id, spec.config_filename, revision=desired_rev, force_download=force
        )
    except EntryNotFoundError:
        config_path = None

    st.update({
        "repo_id": spec.repo_id,
        "weights": spec.weights_filename,
        "config": spec.config_filename,
        "revision": desired_rev,
        "weights_path": weights_path,
        "config_path": config_path,
    })
    save_state(st)

    if not weights_path or not Path(weights_path).is_file():
        raise RuntimeError("Weights file could not be resolved/downloaded.")
    return weights_path, config_path


RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"


def latest_release_info(timeout: float = 3.0) -> Optional[dict]:
    """Tag, title, notes and URL of the newest published release.

    None when offline, rate-limited, or when the repository has no published
    release. A pushed git tag is not enough; the API only reports Releases.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    tag = data.get("tag_name")
    if not tag:
        return None
    return {
        "tag": tag,
        "title": (data.get("name") or tag).strip(),
        "notes": (data.get("body") or "").strip(),
        "url": data.get("html_url") or RELEASES_URL,
    }


def latest_app_release(timeout: float = 3.0) -> Optional[str]:
    """Newest published release tag on GitHub, or None if it cannot be reached."""
    info = latest_release_info(timeout)
    return info["tag"] if info else None


def app_update_available() -> Optional[dict]:
    """Release info when a newer release exists, else None. Safe offline.

    Dev builds are excluded: `parse_semver_tag` rejects prereleases, and a
    `.devN` version is one, so a working copy is never nagged.
    """
    info = latest_release_info()
    if not info:
        return None
    latest, current = parse_semver_tag(info["tag"]), parse_semver_tag(__version__)
    if latest is None or current is None:
        return None
    return info if latest > current else None


def update_instructions() -> str:
    """How to install the update, for the way this copy was installed.

    Deliberately returns commands rather than running them: the app is executing
    from the very files an update would replace, and a `git pull` can collide
    with a user's local changes. Updating is the user's call, not ours.
    """
    repo = Path(__file__).resolve().parents[2]
    if (repo / ".git").exists():
        return f'git -C "{repo}" pull\npip install -e "{repo}"'
    return f"pip install --upgrade {APP_NAME}"


def provenance(config_path: Optional[str]) -> dict:
    """Identify the code and the weights behind a result, for reproducibility."""
    cfg = {}
    if config_path and os.path.isfile(config_path):
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    st = load_state()
    return {
        "app_version": __version__,
        "model_repo": st.get("repo_id", HF_REPO_ID),
        "model_revision": st.get("revision", "unknown"),
        "model_version": cfg.get("model_version", "unknown"),
        "arch_version": cfg.get("arch_version", "unknown"),
    }


# ----------------------------- I/O -----------------------------
def normalize_float01(vol: np.ndarray) -> np.ndarray:
    """Map a volume to [0,1] without collapsing its dynamic range.

    Returns the input untouched when it is already float32 in range, so callers
    can normalize defensively without paying a full-volume copy each time.
    """
    x0 = np.asarray(vol)
    if x0.ndim == 2:
        x0 = x0[None, ...]
    if x0.ndim != 3:
        raise ValueError(f"Expected 3D or 2D array; got shape {x0.shape}")

    x = x0.astype(np.float32, copy=False)

    if np.issubdtype(x0.dtype, np.integer):
        maxv = float(np.iinfo(x0.dtype).max)
        if maxv <= 0:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip(x / maxv, 0.0, 1.0)

    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0.0 and vmax <= 1.0:
        return x  # already normalized; copying it again is pure waste
    if vmin >= 0.0 and vmax <= 1.5:
        return np.clip(x, 0.0, 1.0)

    lo, hi = np.percentile(x, [1.0, 99.9])
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        span = max(vmax - vmin, 1.0)
        lo, hi = vmin, vmin + span
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _read_dir_tif_stack(dirpath: Path) -> np.ndarray:
    files: List[Path] = sorted(
        [p for p in dirpath.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No .tif/.tiff files found in: {dirpath}")
    return normalize_float01(np.stack([tiff.imread(str(p)) for p in files], axis=0))


def read_volume_auto(path: Path) -> np.ndarray:
    """Read a .tif/.tiff stack, a .npy array, or a directory of TIFF slices."""
    path = Path(path)
    if path.is_dir():
        return _read_dir_tif_stack(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return read_volume_float01(str(path))
    if ext == ".npy":
        return normalize_float01(np.load(str(path)))
    raise ValueError(f"Unsupported input: {path}")


def write_volume(path: Path, vol: np.ndarray, dtype: str = "uint16"):
    """Write a [0,1] float volume as a TIFF stack or .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(str(path), vol.astype(np.float32, copy=False))
        return
    if dtype == "float32":
        tiff.imwrite(str(path), vol.astype(np.float32, copy=False))
    elif dtype == "uint8":
        tiff.imwrite(str(path), (np.clip(vol, 0, 1) * 255.0).round().astype(np.uint8))
    else:
        tiff.imwrite(str(path), (np.clip(vol, 0, 1) * 65535.0).round().astype(np.uint16))


# ----------------------------- Model -----------------------------
@lru_cache(maxsize=2)
def cached_model(weights_path: str, config_path: Optional[str], device: str):
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    if not config_path or not os.path.isfile(config_path):
        raise RuntimeError("config.json is missing alongside the weights (expected in HF repo).")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    Model = UNet3D_Residual

    params = set(inspect.signature(Model.__init__).parameters.keys())

    def m(names, val):
        for n in names:
            if n in params:
                return {n: val}
        return {}

    kw = {}
    kw |= m(["in_ch", "in_channels", "n_channels", "channels", "input_channels"],
            int(cfg.get("in_channels", 1)))
    kw |= m(["out_ch", "out_channels", "n_classes", "classes", "num_classes"],
            int(cfg.get("out_channels", 1)))
    kw |= m(["base_ch", "base", "features", "width", "base_filters"],
            int(cfg.get("base_channels", 16)))
    kw |= m(["levels", "depth", "num_levels", "n_levels"], int(cfg.get("levels", 4)))

    try:
        net = Model(**{k: v for k, v in kw.items() if k != "self"})
    except TypeError:
        kw2 = {k: v for k, v in kw.items()
               if k not in {"out_ch", "out_channels", "n_classes", "classes", "num_classes"}}
        try:
            net = Model(**kw2)
        except TypeError:
            net = Model()

    net.load_state_dict(load_safetensors(weights_path, device="cpu"), strict=True)
    net.to(dev).eval()
    return net, dev


# ----------------------------- Residual cache -----------------------------
# Entries hold two full-volume tensors, so they are kept in host memory and the
# cache is capped: an unbounded GPU-resident cache pinned VRAM for every volume
# the user had ever run.
_RES_CACHE: dict[tuple, dict] = {}
_RES_CACHE_MAXSIZE = 1


def _fingerprint(arr: np.ndarray) -> tuple:
    arr = np.asarray(arr)
    return (arr.shape, str(arr.dtype), int(arr.nbytes), float(arr.mean()), float(arr.std()))


def _cache_key(weights_path: str, revision: str, device: str, vol: np.ndarray,
               **inference: object) -> tuple:
    """Identify a cached residual.

    Everything that changes the network's output belongs here. The control
    parameters deliberately do not: reusing one residual across control sweeps
    is the entire point of the cache. Tiling was missing, so switching preset
    silently reused the residual computed with the previous tiling grid.
    """
    settings = tuple(sorted((k, repr(v)) for k, v in inference.items()))
    return (weights_path, revision, device, settings, *_fingerprint(vol))


def _cache_store(key: tuple, vol_t: torch.Tensor, r_t: torch.Tensor):
    while len(_RES_CACHE) >= _RES_CACHE_MAXSIZE:
        _RES_CACHE.pop(next(iter(_RES_CACHE)))
    _RES_CACHE[key] = {"vol_t": vol_t, "residual_t": r_t}


def clear_residual_cache():
    _RES_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------- Controls -----------------------------
# Target voxels per slab. Bounds the device working set of the control step so it
# stays O(slab) instead of O(volume).
_CTRL_SLAB_VOXELS = 32_000_000


@torch.no_grad()
def apply_controls_slabwise(
    ctrl: ControlledUNet3D,
    vol_t: torch.Tensor,
    r_t: torch.Tensor,
    dev: str,
    *,
    strength: float,
    hp_sigma: float,
    hp_gain: float,
    lp_gain: float,
) -> np.ndarray:
    """Apply the control formula in Z-slabs, moving one slab at a time to `dev`.

    Slabs are read back with a halo matching the Gaussian kernel radius used by
    gaussian_blur3d, so the result is identical to a whole-volume application.
    """
    D, H, W = int(vol_t.shape[2]), int(vol_t.shape[3]), int(vol_t.shape[4])
    out = np.empty((D, H, W), dtype=np.float32)

    halo = max(1, int(math.ceil(3.0 * hp_sigma))) if hp_sigma and hp_sigma > 0 else 0
    slab = max(1, min(D, int(_CTRL_SLAB_VOXELS // max(1, H * W))))

    for z0 in range(0, D, slab):
        z1 = min(D, z0 + slab)
        a, b = max(0, z0 - halo), min(D, z1 + halo)
        x = vol_t[:, :, a:b].to(dev, non_blocking=True)
        r = r_t[:, :, a:b].to(dev, non_blocking=True)
        y = ctrl.apply_controls(
            x=x, r=r, strength=strength, hp_sigma=hp_sigma,
            hp_gain=hp_gain, lp_gain=lp_gain,
        )
        out[z0:z1] = y[0, 0, z0 - a: z1 - a].to("cpu").numpy()
        del x, r, y

    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def run_inference(
    vol_f32_01: np.ndarray,
    *,
    device: str,
    tile: Tuple[int, int, int],
    overlap: Tuple[int, int, int],
    weights_path: str,
    config_path: Optional[str],
    use_amp: Union[bool, str] = "auto",
    pad_mode: str = "reflect",
    clamp01: bool = True,
    strength: float = 1.0,
    hp_sigma: float = 0.0,
    hp_gain: float = 1.0,
    lp_gain: float = 1.0,
    reuse_cache: bool = True,
    batch_size: Union[int, str] = "auto",
    border_margin: Union[int, str] = "auto",
    progress: Optional[Callable[[int, int], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Run the network once (tiled) and apply the inference-time controls.

    The residual is cached, so repeated calls that differ only in the control
    parameters skip the network entirely.
    """
    validate_volume_shape(np.asarray(vol_f32_01).shape)
    base, dev = cached_model(weights_path, config_path, device)
    vol_f32_01 = normalize_float01(np.asarray(vol_f32_01))

    revision = load_state().get("revision", "unknown")
    key = _cache_key(
        weights_path, revision, dev, vol_f32_01,
        tile=tuple(tile), overlap=tuple(overlap), pad_mode=pad_mode,
        clamp01=clamp01, use_amp=use_amp, batch_size=batch_size,
        border_margin=border_margin,
    )
    ctrl = ControlledUNet3D(base, clamp01=clamp01).eval()

    if reuse_cache and key in _RES_CACHE:
        entry = _RES_CACHE[key]
        if progress is not None:
            progress(1, 1)
        return apply_controls_slabwise(
            ctrl, entry["vol_t"], entry["residual_t"], dev,
            strength=strength, hp_sigma=hp_sigma, hp_gain=hp_gain, lp_gain=lp_gain,
        )

    pred_base = deblur_volume_tiled(
        net=base,
        vol=vol_f32_01,
        tile=tile, overlap=overlap,
        device=dev, use_amp=use_amp,
        pad_mode=pad_mode, clamp01=clamp01,
        batch_size=batch_size,
        border_margin=border_margin,
        progress=progress,
        should_abort=should_abort,
    )

    D, H, W = pred_base.shape
    # Kept on the host: two full-volume tensors per entry would otherwise sit in
    # VRAM for the lifetime of the session.
    vol_t = torch.from_numpy(vol_f32_01).reshape(1, 1, D, H, W)
    base_t = torch.from_numpy(pred_base).reshape(1, 1, D, H, W)
    r_t = base_t - vol_t
    del base_t
    _cache_store(key, vol_t, r_t)

    return apply_controls_slabwise(
        ctrl, vol_t, r_t, dev,
        strength=strength, hp_sigma=hp_sigma, hp_gain=hp_gain, lp_gain=lp_gain,
    )


__all__ = [
    "HFModelSpec", "HF_REPO_ID", "HF_FILENAME", "TILE_PRESETS", "DEFAULT_PRESET",
    "InferenceAborted", "apply_controls_slabwise", "app_update_available",
    "cached_model", "clear_residual_cache", "ensure_model_assets", "has_internet",
    "latest_app_release", "load_state", "normalize_float01", "provenance",
    "read_volume_auto", "run_inference", "save_state", "write_volume",
]
