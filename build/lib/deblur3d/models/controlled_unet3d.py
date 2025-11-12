import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ControlledUNet3D", "gaussian_blur3d"]

# ============================================================
# =============== Gaussian blur utilities ==================== 
# ============================================================

@torch.no_grad()
def _gauss1d(sigma: float, device, dtype=torch.float32, radius_mult: float = 3.0):
    sigma = max(1e-6, float(sigma))
    r = int(math.ceil(radius_mult * sigma))
    r = max(r, 1)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / (k.sum() + 1e-12)
    return k, r


@torch.no_grad()
def gaussian_blur3d(x: torch.Tensor, sigma: float, pad_mode: str = "replicate") -> torch.Tensor:
    """Separable 3D Gaussian blur for (N,C,D,H,W)."""
    if sigma <= 0:
        return x
    k1d, r = _gauss1d(sigma, x.device, x.dtype)
    kz = k1d.view(1, 1, -1, 1, 1)
    ky = k1d.view(1, 1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, 1, -1)
    # depthwise separable filtering
    y = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode=pad_mode), kz, groups=x.shape[1])
    y = F.conv3d(F.pad(y, (0, 0, r, r, 0, 0), mode=pad_mode), ky, groups=x.shape[1])
    y = F.conv3d(F.pad(y, (r, r, 0, 0, 0, 0), mode=pad_mode), kx, groups=x.shape[1])
    return y


# ============================================================
# === Controlled U-Net wrapper with cache-reuse capability ===
# ============================================================

class ControlledUNet3D(nn.Module):
    """
    Wraps a residual UNet (predicts y ≈ clamp(x + resid)) and exposes inference-time controls:
      - strength : global residual scale (0=identity, 1=original, >1=stronger)
      - hp_sigma : Gaussian σ (vox) to split residual into low/high freq
      - hp_gain  : scale factor for high-frequency part
      - lp_gain  : scale factor for low-frequency part

    The wrapper supports *two usage modes*:

      1. **Direct mode**  →  y_ctrl = ctrl(x, ...)
         Runs the UNet once each call (backward-compatible).

      2. **Cached mode**  →  r = ctrl.compute_residual(x); then ctrl.apply_controls(x, r, ...)
         Allows reusing the same residual for multiple control sweeps (fast).
    """

    def __init__(self, net: nn.Module, clamp01: bool = True):
        super().__init__()
        self.net = net
        self.clamp01 = clamp01
        self._cached = None  # optional (x, r) cache

    # --------------------------------------------------------
    #  Mode 2 — explicit separation for reuse
    # --------------------------------------------------------
    @torch.no_grad()
    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Run the UNet once and return residual r = y - x (cached)."""
        y = self.net(x)
        r = y - x
        self._cached = (x, r)
        return r

    @torch.no_grad()
    def apply_controls(
        self,
        x: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        *,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
        pad_mode: str = "replicate",
    ) -> torch.Tensor:
        """
        Apply frequency-based control on a precomputed residual (VRAM-light).

        If x/r are None, uses the last cached pair from compute_residual().
        """
        if r is None or x is None:
            if self._cached is None:
                raise RuntimeError("No cached residual. Call compute_residual(x) first.")
            x, r = self._cached

        # Split residual into low/high frequencies
        if hp_sigma and hp_sigma > 0:
            r_lp = gaussian_blur3d(r, sigma=hp_sigma, pad_mode=pad_mode)
            r_hp = r - r_lp
            r_mod = lp_gain * r_lp + hp_gain * r_hp
        else:
            r_mod = r

        y_ctrl = x + float(strength) * r_mod
        if self.clamp01:
            y_ctrl = y_ctrl.clamp(0, 1)
        return y_ctrl

    # --------------------------------------------------------
    #  Mode 1 — legacy direct forward (still supported)
    # --------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
    ) -> torch.Tensor:
        """
        Compatibility path that internally calls compute_residual+apply_controls.
        """
        r = self.compute_residual(x)
        return self.apply_controls(x, r, strength=strength, hp_sigma=hp_sigma,
                                   hp_gain=hp_gain, lp_gain=lp_gain)
