# src/deblur3d/app/gui.py
from __future__ import annotations

import os
import json
import socket
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Optional
from packaging.version import Version, InvalidVersion

import numpy as np
try:
    import torch  # noqa
except Exception as e:
    raise RuntimeError(
        "PyTorch is not installed. Install CPU: `pip install -e .[cpu]` "
        "or CUDA 11.6: `pip install -e .[cu116] --extra-index-url https://download.pytorch.org/whl/cu116`"
    ) from e

import tifffile as tiff

from magicgui import magicgui
from napari import Viewer, run
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtWidgets import QMessageBox
from platformdirs import user_data_dir
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
except Exception:
    try:
        from huggingface_hub import RepositoryNotFoundError, EntryNotFoundError  # type: ignore
    except Exception:
        class RepositoryNotFoundError(Exception): ...
        class EntryNotFoundError(Exception): ...

from ._workers import make_infer_worker

# ---- Project imports ----
from deblur3d.data.io import read_volume_float01
from deblur3d.infer.tiled import deblur_volume_tiled
from deblur3d.models import UNet3D_Residual, ControlledUNet3D

# =========================
# Hugging Face integration
# =========================
HF_REPO_ID = os.getenv("DEBLUR3D_HF_REPO", "HippoCanFly/DeepDeBlur3D")
HF_FILENAME = os.getenv("DEBLUR3D_HF_FILE", "pytorch_model.safetensors")
HF_REVISION_DEFAULT = os.getenv("DEBLUR3D_HF_REV", "v1.0.0")

APP_AUTHOR = "DeepDeBlur3D"
APP_NAME   = "deblur3d-gui"
STATE_DIR  = Path(user_data_dir(APP_NAME, APP_AUTHOR)); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "model_state.json"

@dataclass
class HFModelSpec:
    repo_id: str
    weights_filename: str
    config_filename: str = "config.json"
    revision: Optional[str] = None

def _has_internet(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout)
        return True
    except OSError:
        return False

def _ask_yes_no(title: str, text: str) -> bool:
    m = QMessageBox()
    m.setIcon(QMessageBox.Question)
    m.setWindowTitle(title)
    m.setText(text)
    m.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    m.setDefaultButton(QMessageBox.Yes)
    return m.exec_() == QMessageBox.Yes

def _load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_state(d: dict):
    try:
        STATE_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def _parse_semver_tag(tag: str) -> Optional[Version]:
    s = tag[1:] if tag.startswith("v") else tag
    try:
        v = Version(s)
        return v if not v.is_prerelease else None
    except InvalidVersion:
        return None

def _get_latest_semver_tag(api: HfApi, repo_id: str) -> Optional[str]:
    refs = api.list_repo_refs(repo_id)
    best: tuple[Version, str] | None = None
    for t in refs.tags:
        v = _parse_semver_tag(t.name)
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, t.name)
    return best[1] if best else None

def _get_desired_revision_by_tag(api: HfApi, repo_id: str) -> str:
    state = _load_state()
    current = state.get("revision") or HF_REVISION_DEFAULT
    cur_ver = _parse_semver_tag(current)
    try:
        latest = _get_latest_semver_tag(api, repo_id)
    except Exception:
        latest = None
    if latest:
        lat_ver = _parse_semver_tag(latest)
        if cur_ver and lat_ver and lat_ver > cur_ver:
            if _ask_yes_no(
                "Model update available",
                f"A newer tagged model ({latest}) is available.\n\n"
                f"Update from {current} to {latest}?"
            ):
                current = latest
                state["revision"] = current
                _save_state(state)
    if state.get("revision") != current:
        state["revision"] = current
        _save_state(state)
    return current

def ensure_model_assets(spec: HFModelSpec) -> Tuple[str, Optional[str]]:
    api = HfApi()
    online = _has_internet()
    if online:
        desired_rev = _get_desired_revision_by_tag(api, spec.repo_id)
    else:
        st = _load_state()
        desired_rev = st.get("revision") or HF_REVISION_DEFAULT

    def _download_all(force=False):
        w = hf_hub_download(spec.repo_id, spec.weights_filename, revision=desired_rev, force_download=force)
        c = None
        try:
            c = hf_hub_download(spec.repo_id, spec.config_filename, revision=desired_rev, force_download=force)
        except EntryNotFoundError:
            c = None
        return w, c

    st = _load_state()
    last_rev = st.get("revision")
    force = online and (last_rev is not None) and (desired_rev != last_rev)

    weights_path, config_path = _download_all(force=force)

    st.update({
        "repo_id": spec.repo_id,
        "weights": spec.weights_filename,
        "config": spec.config_filename,
        "revision": desired_rev,
        "weights_path": weights_path,
        "config_path": config_path,
    })
    _save_state(st)

    if not weights_path or not Path(weights_path).is_file():
        raise RuntimeError("Weights file could not be resolved/downloaded.")
    return weights_path, config_path

# ----------------- I/O + normalization -----------------
def _normalize_float01_like_io(vol: np.ndarray) -> np.ndarray:
    """
    Robust mapping to [0,1] that avoids collapsing dynamic range (prevents single-handle slider).
    """
    x0 = np.asarray(vol)
    if x0.ndim == 2:
        x0 = x0[None, ...]
    if x0.ndim != 3:
        raise ValueError(f"Expected 3D or 2D array; got shape {x0.shape}")

    # Always float32 internally
    x = x0.astype(np.float32, copy=False)

    # Integer input: scale by dtype max
    if np.issubdtype(x0.dtype, np.integer):
        maxv = float(np.iinfo(x0.dtype).max)
        if maxv <= 0:
            return np.zeros_like(x, dtype=np.float32)
        x = x / maxv
        return np.clip(x, 0.0, 1.0)

    # Float input
    vmin, vmax = float(x.min()), float(x.max())
    if vmin >= 0.0 and vmax <= 1.5:
        # Already normalized-ish → only clip
        return np.clip(x, 0.0, 1.0)

    # Percentile remap with stability guards
    lo, hi = np.percentile(x, [1.0, 99.9])
    # If degenerate or NaN, fall back to a non-collapsing span
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        span = max(vmax - vmin, 1.0)  # ensure non-zero
        lo, hi = vmin, vmin + span

    x = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(x, 0.0, 1.0)

def _read_dir_tif_stack(dirpath: Path) -> np.ndarray:
    files: List[Path] = sorted(
        [p for p in dirpath.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No .tif/.tiff files found in: {dirpath}")
    vol = np.stack([tiff.imread(str(p)) for p in files], axis=0)
    return _normalize_float01_like_io(vol)

def read_volume_auto(path: Path) -> np.ndarray:
    if path.is_dir():
        return _read_dir_tif_stack(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return read_volume_float01(str(path))
    if ext == ".npy":
        arr = np.load(str(path))
        return _normalize_float01_like_io(arr)
    raise ValueError(f"Unsupported input: {path}")

# ----------------- Model cache/loader -----------------
import inspect

@lru_cache(maxsize=2)
def _cached_model_from_paths(weights_path: str, config_path: Optional[str], device: str):
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    if not config_path or not os.path.isfile(config_path):
        raise RuntimeError("config.json is missing alongside the weights (expected in HF repo).")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    Model = UNet3D_Residual

    sig = inspect.signature(Model.__init__)
    params = set(sig.parameters.keys())
    def m(names, val):
        for n in names:
            if n in params:
                return {n: val}
        return {}
    kw = {}
    kw |= m(["in_ch","in_channels","n_channels","channels","input_channels"], int(cfg.get("in_channels", 1)))
    out_val = int(cfg.get("out_channels", 1))
    kw |= m(["out_ch","out_channels","n_classes","classes","num_classes"], out_val)
    kw |= m(["base_ch","base","features","width","base_filters"], int(cfg.get("base_channels", 16)))
    kw |= m(["levels","depth","num_levels","n_levels"], int(cfg.get("levels", 4)))

    try:
        net = Model(**{k: v for k, v in kw.items() if k != "self"})
    except TypeError:
        kw2 = {k: v for k, v in kw.items() if k not in {"out_ch","out_channels","n_classes","classes","num_classes"}}
        try:
            net = Model(**kw2)
        except TypeError:
            net = Model()

    sd = load_safetensors(weights_path, device="cpu")
    net.load_state_dict(sd, strict=True)
    net.to(dev).eval()
    return net, dev

# ----------------- Residual cache -----------------
_RES_CACHE: dict[tuple, dict] = {}

def _fingerprint(arr: np.ndarray) -> tuple:
    arr = np.asarray(arr)
    return (arr.shape, str(arr.dtype), int(arr.nbytes), float(arr.mean()), float(arr.std()))

def _cache_key(weights_path: str, revision: str, device: str, vol: np.ndarray) -> tuple:
    return (weights_path, revision, device, *_fingerprint(vol))

def clear_residual_cache():
    _RES_CACHE.clear()
    show_info("DeepDeBlur3D residual cache cleared.")

# ----------------- Contrast stabilization -----------------
def _stabilize_contrast(layer: NapariImage, lo: float = 0.0, hi: float = 1.0):
    """
    Set both contrast_limits_range and contrast_limits to a sane (lo, hi),
    ensuring a two-handle slider and avoiding degenerate domains.
    """
    try:
        layer.contrast_limits_range = (float(lo), float(hi))
    except Exception:
        pass
    layer.contrast_limits = (float(lo), float(hi))
    # mark prepared
    if getattr(layer, "metadata", None) is not None:
        layer.metadata["deblur3d_prepared"] = True

# ----------------- Direct-mode controlled wrapper -----------------
class _ParamNetDirect(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, clamp01: bool,
                 strength: float, hp_sigma: float, hp_gain: float, lp_gain: float):
        super().__init__()
        self.ctrl = ControlledUNet3D(base, clamp01=clamp01)
        self.strength = float(strength)
        self.hp_sigma = float(hp_sigma)
        self.hp_gain = float(hp_gain)
        self.lp_gain = float(lp_gain)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctrl(
            x,
            strength=self.strength,
            hp_sigma=self.hp_sigma,
            hp_gain=self.hp_gain,
            lp_gain=self.lp_gain,
        )

# ----------------- Inference wrapper (with residual reuse) -----------------
def run_infer_bound(
    vol_f32_01: np.ndarray,
    *,
    device: str,
    tile: Tuple[int, int, int],
    overlap: Tuple[int, int, int],
    use_amp: bool = False,
    pad_mode: str = "reflect",
    clamp01: bool = True,
    strength: float = 1.0,
    hp_sigma: float = 0.0,
    hp_gain: float = 1.0,
    lp_gain: float = 1.0,
    weights_path: str,
    config_path: Optional[str],
    reuse_cache: bool = True,
) -> np.ndarray:
    base, dev = _cached_model_from_paths(weights_path, config_path, device)
    vol_f32_01 = _normalize_float01_like_io(np.asarray(vol_f32_01))

    st = _load_state()
    revision = st.get("revision", "unknown")
    key = _cache_key(weights_path, revision, dev, vol_f32_01)

    # Fast path: reuse cached residual
    if reuse_cache and key in _RES_CACHE:
        entry = _RES_CACHE[key]
        vol_t: torch.Tensor = entry["vol_t"]      # [1,1,D,H,W]
        r_t:   torch.Tensor = entry["residual_t"] # [1,1,D,H,W]
        ctrl = ControlledUNet3D(base, clamp01=clamp01).to(dev).eval()
        with torch.no_grad():
            y_ctrl = ctrl.apply_controls(x=vol_t, r=r_t,
                                         strength=strength, hp_sigma=hp_sigma,
                                         hp_gain=hp_gain, lp_gain=lp_gain)
        return y_ctrl.squeeze(0).squeeze(0).detach().to("cpu").numpy().astype(np.float32, copy=False)

    # Cache miss → run UNet once (tiled) for base output; apply controls after
    pred_base = deblur_volume_tiled(
        net=base,
        vol=vol_f32_01,
        tile=tile, overlap=overlap,
        device=dev, use_amp=use_amp if dev == "cuda" else False,
        pad_mode=pad_mode, clamp01=clamp01,
    )

    D, H, W = pred_base.shape
    vol_t = torch.from_numpy(vol_f32_01).to(dev, dtype=torch.float32).view(1,1,D,H,W)
    base_t = torch.from_numpy(pred_base).to(dev, dtype=torch.float32).view(1,1,D,H,W)
    r_t    = base_t - vol_t
    _RES_CACHE[key] = {"vol_t": vol_t, "residual_t": r_t}

    ctrl = ControlledUNet3D(base, clamp01=clamp01).to(dev).eval()
    with torch.no_grad():
        y_ctrl = ctrl.apply_controls(x=vol_t, r=r_t,
                                     strength=strength, hp_sigma=hp_sigma,
                                     hp_gain=hp_gain, lp_gain=lp_gain)
    return y_ctrl.squeeze(0).squeeze(0).detach().to("cpu").numpy().astype(np.float32, copy=False)

# ----------------- Napari GUI -----------------
def build_viewer() -> Viewer:
    v = Viewer(title="deblur3d — Inference")
    v.dims.ndisplay = 2
    v.grid.enabled = True

    state = {"run_idx": 1}

    # One-time input prep with stabilized contrast
    def _prepare_input_layer(layer: NapariImage):
        if getattr(layer, "metadata", None) and layer.metadata.get("deblur3d_prepared"):
            return True

        data = np.asarray(layer.data)
        if data.ndim not in (2, 3):
            return False
        try:
            norm = _normalize_float01_like_io(data)
        except Exception:
            return False

        layer.data = norm.astype(np.float32, copy=False)
        layer.colormap = "gray"
        _stabilize_contrast(layer, 0.0, 1.0)  # critical for two-handle slider

        v.dims.ndisplay = 2
        v.grid.enabled = True
        return True

    def _update_run_enabled_from_active():
        active = v.layers.selection.active
        enable = isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) in (2, 3)
        infer_w.enabled = bool(enable)
        clear_cache_btn.enabled = True

    # Only treat user inputs; ignore app-generated outputs
    def _on_layer_added(event):
        layer = event.value
        if not isinstance(layer, NapariImage):
            return
        if getattr(layer, "metadata", None) and layer.metadata.get("deblur3d_output"):
            return
        if _prepare_input_layer(layer):
            v.layers.selection.active = layer
            _update_run_enabled_from_active()
            show_info(f"Input set from drag-and-drop: {layer.name}")

    v.layers.events.inserted.connect(_on_layer_added)
    v.layers.selection.events.active.connect(lambda e: _update_run_enabled_from_active())

    @magicgui(call_button="Clear residual cache")
    def clear_cache_btn():
        clear_residual_cache()

    @magicgui(
        call_button="Run Filter",
        device={"choices": ["cuda", "cpu"]},
        tile_x={"label": "Tile X", "min": 16, "max": 512, "step": 16, "value": 256},
        tile_y={"label": "Tile Y", "min": 16, "max": 512, "step": 16, "value": 256},
        tile_z={"label": "Tile Z", "min": 8,  "max": 128, "step": 8,  "value": 64},
        ov_x={"label": "Overlap X", "min": 0, "max": 256, "step": 8, "value": 128},
        ov_y={"label": "Overlap Y", "min": 0, "max": 256, "step": 8, "value": 128},
        ov_z={"label": "Overlap Z", "min": 0, "max": 64,  "step": 4, "value": 32},
        use_amp={"label": "Use AMP", "value": False},
        pad_mode={"choices": ["reflect", "replicate", "constant"], "value": "reflect"},
        clamp01={"label": "Clamp [0,1]", "value": True},
        strength={"label": "Strength", "min": 0.0, "max": 3.0, "step": 0.1, "value": 1.0},
        hp_sigma={"label": "HP Sigma (vox)", "min": 0.0, "max": 8.0, "step": 0.1, "value": 0.0},
        hp_gain={"label": "HP Gain", "min": 0.0, "max": 4.0, "step": 0.1, "value": 1.0},
        lp_gain={"label": "LP Gain", "min": 0.0, "max": 4.0, "step": 0.1, "value": 1.0},
        reuse_cache={"label": "Reuse cached residual (fast)", "value": True},
    )
    def infer_w(
        device: str = "cuda",
        tile_x: int = 256, tile_y: int = 256, tile_z: int = 64,
        ov_x: int = 128,  ov_y: int = 128,  ov_z: int = 32,
        use_amp: bool = False,
        pad_mode: str = "reflect",
        clamp01: bool = True,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
        reuse_cache: bool = True,
    ):
        def _is_cuda_error(err: Exception) -> bool:
            msg = str(err).lower()
            return ("cuda" in msg or "cudnn" in msg or "device-side assert" in msg)

        active = v.layers.selection.active
        if not (isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) in (2, 3)):
            show_warning("Select an image layer (2D/3D) as input.")
            return

        vol = _normalize_float01_like_io(np.asarray(active.data))
        tile = (tile_z, tile_y, tile_x)   # (Z, Y, X)
        overlap = (ov_z, ov_y, ov_x)

        try:
            desired_spec = HFModelSpec(repo_id=HF_REPO_ID, weights_filename=HF_FILENAME, revision=None)
            weights_path, config_path = ensure_model_assets(desired_spec)
        except Exception as e:
            show_error(f"Model resolution failed: {e}")
            return

        want_cuda = (device == "cuda")
        cuda_available = (torch.cuda.is_available() if want_cuda else False)
        first_device = "cuda" if (want_cuda and cuda_available) else "cpu"
        if want_cuda and not cuda_available:
            show_warning("CUDA not available. Falling back to CPU.")
        print(f"[DeepDeBlur3D] Using device: {first_device}")

        infer_w.enabled = False
        run_id = state["run_idx"]
        start = None

        def _fmt(x: float, nd=2):
            return f"{x:.1f}" if nd == 1 else f"{x:.2f}"

        def _launch(device_to_use: str):
            nonlocal start
            start = time.time()
            show_info(f"Starting inference on '{active.name}' using {device_to_use.upper()} …")
            print(f"[DeepDeBlur3D] Starting inference on '{active.name}' using {device_to_use}")

            worker = make_infer_worker(
                lambda v_, device=None, progress=None: run_infer_bound(
                    v_,
                    device=device_to_use,
                    tile=tile,
                    overlap=overlap,
                    use_amp=use_amp if device_to_use == "cuda" else False,
                    pad_mode=pad_mode,
                    clamp01=clamp01,
                    strength=strength,
                    hp_sigma=hp_sigma,
                    hp_gain=hp_gain,
                    lp_gain=lp_gain,
                    weights_path=weights_path,
                    config_path=config_path,
                    reuse_cache=reuse_cache,
                ),
                vol, device=device_to_use, extra_kwargs={}
            )

            def on_return(pred: np.ndarray):
                dt = time.time() - start if start else 0.0
                layer_name = (
                    f"filtered_s{_fmt(strength,1)}_"
                    f"hps{_fmt(hp_sigma,2)}_hpg{_fmt(hp_gain,2)}_lpg{_fmt(lp_gain,2)}_"
                    f"{run_id}"
                )
                lyr = v.add_image(
                    pred, name=layer_name, colormap="magenta",
                    blending="additive", opacity=0.7
                )
                # tag and stabilize contrast so slider behaves correctly
                if getattr(lyr, "metadata", None) is not None:
                    lyr.metadata["deblur3d_output"] = True
                _stabilize_contrast(lyr, 0.0, 1.0)

                lyr.grid_position = (0, 1)
                show_info(f"Inference #{run_id} done in {dt:.2f}s on {device_to_use.upper()} | shape={pred.shape}")
                print(f"[DeepDeBlur3D] Inference #{run_id} done in {dt:.2f}s on {device_to_use}")
                v.grid.enabled = True
                state["run_idx"] = run_id + 1
                infer_w.enabled = True

            def on_error(e):
                msg = str(e)
                print(f"[DeepDeBlur3D] ERROR on {device_to_use}: {msg}")
                if device_to_use == "cuda" and _is_cuda_error(e):
                    show_warning("CUDA run failed. Falling back to CPU automatically…")
                    print("[DeepDeBlur3D] Falling back to CPU due to CUDA error")
                    _launch("cpu")
                else:
                    infer_w.enabled = True
                    show_error(f"Inference error on {device_to_use}: {e}")

            worker.returned.connect(on_return)
            worker.errored.connect(on_error)
            worker.start()

        _launch(first_device)

    # Widgets
    infer_w.enabled = False
    v.window.add_dock_widget(infer_w, name="DeepDeBlur3D", area="right")
    v.window.add_dock_widget(clear_cache_btn, name="Cache", area="right")
    return v

def main():
    v = build_viewer()
    run()

if __name__ == "__main__":
    main()
