"""Headless command line interface for DeepDeBlur3D.

    deblur3d scan.tif out.tif
    deblur3d scans/ out/ --batch --strength 1.2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from . import __version__
from .core import (
    DEFAULT_PRESET,
    app_update_available,
    HF_FILENAME,
    HF_REPO_ID,
    TILE_PRESETS,
    HFModelSpec,
    InferenceAborted,
    ensure_model_assets,
    provenance,
    read_volume_auto,
    run_inference,
    write_volume,
)

VOLUME_SUFFIXES = (".tif", ".tiff", ".npy")
RELEASES_URL = "https://github.com/kiataj/deepdeblur3d-app/releases/latest"


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[deblur3d] CUDA not available; using CPU.", file=sys.stderr)
        return "cpu"
    return requested


def _collect_inputs(root: Path, pattern: str) -> List[Path]:
    """Volumes to process in batch mode: matching files, plus slice directories."""
    if not root.is_dir():
        raise SystemExit(f"--batch needs a directory; {root} is not one.")
    items = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in VOLUME_SUFFIXES
    )
    if not items:
        # A directory of subdirectories, each holding TIFF slices.
        items = sorted(
            d for d in root.iterdir()
            if d.is_dir() and any(c.suffix.lower() in (".tif", ".tiff") for c in d.iterdir())
        )
    if not items:
        raise SystemExit(f"No volumes found under {root} matching {pattern!r}.")
    return items


def _parse_triplet(values: Optional[List[int]], name: str):
    if values is None:
        return None
    if len(values) != 3:
        raise SystemExit(f"--{name} takes exactly three values (Z Y X); got {len(values)}.")
    return tuple(int(v) for v in values)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deblur3d",
        description="Denoise and deblur 3D micro-CT volumes with DeepDeBlur3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", type=Path,
                   help="Volume file, a directory of TIFF slices, or (with --batch) a directory of volumes.")
    p.add_argument("output", nargs="?", type=Path,
                   help="Output file, or output directory when --batch is given.")
    p.add_argument("--batch", action="store_true",
                   help="Treat INPUT as a directory of volumes and OUTPUT as a directory.")
    p.add_argument("--glob", default="*", metavar="PATTERN",
                   help="Which entries to pick up in batch mode.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    g = p.add_argument_group("tiling")
    g.add_argument("--preset", choices=list(TILE_PRESETS), default=DEFAULT_PRESET,
                   help="Named tile/overlap pair.")
    g.add_argument("--tile", nargs=3, type=int, metavar=("Z", "Y", "X"),
                   help="Override the preset's tile size.")
    g.add_argument("--overlap", nargs=3, type=int, metavar=("Z", "Y", "X"),
                   help="Override the preset's overlap.")
    g.add_argument("--tiles-per-pass", default="auto", metavar="N",
                   help="Tiles per forward pass; 'auto' sizes it from free VRAM.")
    g.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (slower, bit-reproducible against older runs).")

    c = p.add_argument_group("controls")
    c.add_argument("--strength", type=float, default=1.0)
    c.add_argument("--hp-sigma", type=float, default=0.0)
    c.add_argument("--hp-gain", type=float, default=1.0)
    c.add_argument("--lp-gain", type=float, default=1.0)

    o = p.add_argument_group("output")
    o.add_argument("--dtype", choices=["uint16", "uint8", "float32"], default="uint16")
    o.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    o.add_argument("--quiet", action="store_true", help="Suppress progress bars.")

    p.add_argument("--list-presets", action="store_true", help="Print tile presets and exit.")
    p.add_argument("--no-update-check", action="store_true",
                   help="Skip the check for a newer release.")
    p.add_argument("--version", action="version", version=f"deblur3d {__version__}")
    return p


def _report_update(enabled: bool):
    """Print a notice if a newer release exists. Never fatal, never slow."""
    if not enabled or os.environ.get("DEBLUR3D_NO_UPDATE_CHECK"):
        return
    try:
        tag = app_update_available()
    except Exception:
        return
    if tag:
        print(
            f"[deblur3d] Update available: {tag} (running {__version__}).\n"
            f"[deblur3d]   {RELEASES_URL}",
            file=sys.stderr,
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    _report_update(not args.no_update_check)

    if args.list_presets:
        for name, (tile, overlap) in TILE_PRESETS.items():
            mark = " (default)" if name == DEFAULT_PRESET else ""
            print(f"{name}{mark}\n    tile={tile}  overlap={overlap}")
        return 0

    if args.input is None or args.output is None:
        build_parser().error("input and output are required")

    tile, overlap = TILE_PRESETS[args.preset]
    tile = _parse_triplet(args.tile, "tile") or tile
    overlap = _parse_triplet(args.overlap, "overlap") or overlap

    try:
        tiles_per_pass = args.tiles_per_pass if args.tiles_per_pass == "auto" else int(args.tiles_per_pass)
    except ValueError:
        raise SystemExit(f"--tiles-per-pass must be an integer or 'auto'; got {args.tiles_per_pass!r}.")

    device = _resolve_device(args.device)

    if args.batch:
        inputs = _collect_inputs(args.input, args.glob)
        args.output.mkdir(parents=True, exist_ok=True)
        suffix = ".npy" if args.dtype == "float32" and args.input.suffix == ".npy" else ".tif"
        outputs = [args.output / f"{p.stem}_deblurred{suffix}" for p in inputs]
    else:
        inputs, outputs = [args.input], [args.output]

    # Resolve the model once. No prompt callback, so a batch run cannot switch
    # model revisions partway through.
    try:
        weights_path, config_path = ensure_model_assets(
            HFModelSpec(repo_id=HF_REPO_ID, weights_filename=HF_FILENAME)
        )
    except Exception as e:
        print(f"[deblur3d] Model resolution failed: {e}", file=sys.stderr)
        return 2

    info = provenance(config_path)
    print(f"[deblur3d] app {info['app_version']} | model {info['model_revision']} "
          f"({info['model_version']}) | device {device} | tile {tile} overlap {overlap}")

    failures = 0
    outer = tqdm(list(zip(inputs, outputs)), desc="volumes", unit="vol",
                 disable=args.quiet or len(inputs) == 1)
    for src, dst in outer:
        if dst.exists() and not args.overwrite:
            print(f"[deblur3d] skip (exists): {dst}. Pass --overwrite to replace.", file=sys.stderr)
            continue
        try:
            vol = read_volume_auto(src)
        except Exception as e:
            print(f"[deblur3d] {src}: could not read ({e})", file=sys.stderr)
            failures += 1
            continue

        bar = tqdm(total=1, desc=src.name, unit="tile", leave=False, disable=args.quiet)

        def on_progress(done: int, total: int, _bar=bar):
            if _bar.total != total:
                _bar.reset(total=total)
            _bar.n = done
            _bar.refresh()

        try:
            result = run_inference(
                vol,
                device=device, tile=tile, overlap=overlap,
                weights_path=weights_path, config_path=config_path,
                use_amp=False if args.no_amp else "auto",
                strength=args.strength, hp_sigma=args.hp_sigma,
                hp_gain=args.hp_gain, lp_gain=args.lp_gain,
                batch_size=tiles_per_pass,
                progress=None if args.quiet else on_progress,
                # Each volume is independent; caching one residual across a batch
                # would only ever hit on a repeat of the same volume.
                reuse_cache=not args.batch,
            )
        except InferenceAborted:
            bar.close()
            print("[deblur3d] aborted.", file=sys.stderr)
            return 130
        except KeyboardInterrupt:
            bar.close()
            print("\n[deblur3d] interrupted.", file=sys.stderr)
            return 130
        except Exception as e:
            bar.close()
            print(f"[deblur3d] {src}: inference failed ({e})", file=sys.stderr)
            failures += 1
            continue
        bar.close()

        try:
            write_volume(dst, result, dtype=args.dtype)
        except Exception as e:
            print(f"[deblur3d] {src}: could not write {dst} ({e})", file=sys.stderr)
            failures += 1
            continue
        if not args.quiet:
            print(f"[deblur3d] {src.name} -> {dst}")

    if failures:
        print(f"[deblur3d] {failures} of {len(inputs)} volume(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
