# src/deblur3d_app/gui.py
"""Napari front end. All inference logic lives in core.py."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    import torch  # noqa
except Exception as e:
    raise RuntimeError(
        "PyTorch is not installed. Install CPU: `pip install -e .[cpu]` "
        "or CUDA 11.6: `pip install -e .[cu116] --extra-index-url https://download.pytorch.org/whl/cu116`"
    ) from e

from magicgui import magicgui
from napari import Viewer, run
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QApplication, QAbstractSpinBox, QDoubleSpinBox, QLabel, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)
from tqdm import tqdm

from . import __version__
from ._workers import make_infer_worker
from .core import (
    DEFAULT_PRESET,
    HF_FILENAME,
    HF_REPO_ID,
    TILE_PRESETS,
    TILE_PRESET_HELP,
    HFModelSpec,
    InferenceAborted,
    app_update_available,
    ensure_model_assets,
    update_instructions,
    normalize_float01,
    provenance,
    run_inference,
)

GITHUB_URL = "https://github.com/kiataj/deepdeblur3d-app/releases/latest"


def _quiet_directwrite_font_warning():
    """Drop one benign Qt warning on Windows; forward every other Qt message.

    napari's dock widgets plus magicgui make Qt resolve a System style-hint font,
    which on Windows is the legacy raster font MS Sans Serif that DirectWrite
    cannot build a face from. Reproducible with napari and magicgui alone, with
    none of this package's code involved, and nothing a user can act on, but it
    printed on every launch.
    """
    if sys.platform != "win32":
        return
    from qtpy.QtCore import qInstallMessageHandler

    def handler(mode, context, message):
        if "CreateFontFaceFromHDC" in message:
            return
        print(message, file=sys.stderr)

    qInstallMessageHandler(handler)


def _ask_yes_no(title: str, text: str) -> bool:
    m = QMessageBox()
    m.setIcon(QMessageBox.Question)
    m.setWindowTitle(title)
    m.setText(text)
    m.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    m.setDefaultButton(QMessageBox.Yes)
    return m.exec_() == QMessageBox.Yes


CONTROL_PARAMS = ("strength", "hp_sigma", "hp_gain", "lp_gain")

# Frame only: no background and no text colour. napari themes the widget through
# its own stylesheet while palette(...) resolves against the Qt palette, so
# setting one of the pair here can land the text on top of a background of the
# same colour, which made these fields render blank until you selected the text.
# A translucent grey border reads on both the light and the dark theme.
_READOUT_STYLE = """
QDoubleSpinBox {
    border: 1px solid rgba(128, 128, 128, 0.55);
    border-radius: 3px;
    padding: 1px 2px;
    min-width: 58px;
}
QDoubleSpinBox:focus { border: 1px solid rgba(110, 165, 255, 0.9); }
"""


def _make_readouts_editable(widget):
    """Give each slider's readout a visible frame and spin arrows.

    superqt's slider readout is already an editable QDoubleSpinBox, but it is
    drawn frameless, so it reads as a static label and users do not realise they
    can type an exact value into it.
    """
    for name in CONTROL_PARAMS:
        sub = getattr(widget, name, None)
        if sub is None:
            continue
        for box in sub.native.findChildren(QDoubleSpinBox):
            box.setFrame(True)
            box.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
            box.setStyleSheet(_READOUT_STYLE)
            box.setToolTip("Type an exact value, or drag the slider.")
            _bind_readout(box, sub)


def _bind_readout(box: QDoubleSpinBox, slider):
    """Push every readout edit into the parameter, not just committed typing.

    superqt's readout only propagates on editingFinished, which typing followed
    by Enter or a focus change does emit. The spin arrows do not, so clicking one
    moved the displayed number while the value used for inference stayed put.
    """
    def on_changed(value: float):
        if slider.value != value:
            slider.value = value

    box.valueChanged.connect(on_changed)


_INFO_BADGE_STYLE = """
QLabel {
    border: 1px solid palette(mid);
    border-radius: 8px;
    min-width: 14px; max-width: 14px;
    min-height: 14px; max-height: 14px;
    font-weight: bold;
    font-size: 10px;
    color: palette(mid);
}
"""


def _preset_tooltip() -> str:
    rows = []
    for name, body in TILE_PRESET_HELP.items():
        lines = "<br>".join(body.splitlines())
        rows.append(f"<p style='margin:0 0 8px 0'><b>{name}</b><br>{lines}</p>")
    return (
        "<div style='max-width:460px'>"
        "<p style='margin:0 0 8px 0'>Tile size sets how the volume is cut up for "
        "the network. It changes both memory use and the result, so it is a fixed "
        "choice rather than something derived from your GPU.</p>"
        + "".join(rows)
        + "</div>"
    )


def _add_preset_info(widget) -> bool:
    """Put an 'i' badge beside the Tiling row and describe each preset on hover."""
    tooltip = _preset_tooltip()
    try:
        combo = widget.preset
        for index, name in enumerate(TILE_PRESET_HELP):
            combo.native.setItemData(index, TILE_PRESET_HELP[name], Qt.ToolTipRole)
        combo.native.setToolTip(tooltip)

        row = combo._labeled_widget()
        layout = row.native.layout()
    except Exception:
        # Private magicgui layout API; a badge is not worth breaking startup over.
        return False

    badge = QLabel("i")
    badge.setAlignment(Qt.AlignCenter)
    badge.setStyleSheet(_INFO_BADGE_STYLE)
    badge.setToolTip(tooltip)
    badge.setCursor(Qt.WhatsThisCursor)
    layout.addWidget(badge)
    return True


def _stabilize_contrast(layer: NapariImage, lo: float = 0.0, hi: float = 1.0):
    """Pin the contrast domain to [0,1] so the slider keeps two usable handles."""
    try:
        layer.contrast_limits_range = (float(lo), float(hi))
    except Exception:
        pass
    layer.contrast_limits = (float(lo), float(hi))
    if getattr(layer, "metadata", None) is not None:
        layer.metadata["deblur3d_prepared"] = True


class _Relay(QObject):
    """Marshals worker-thread events onto the Qt main thread."""
    progressed = Signal(int, int)
    update_found = Signal(dict)
    up_to_date = Signal()


class _ProgressPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.label = QLabel("Idle")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.abort_button = QPushButton("Abort")
        self.abort_button.setEnabled(False)
        self.update_button = QPushButton("Check for updates")
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.abort_button)
        layout.addWidget(self.update_button)

    def set_progress(self, done: int, total: int):
        total = max(1, total)
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.label.setText(f"Tile {done} / {total}")

    def reset(self, message: str = "Idle"):
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.label.setText(message)


def build_viewer() -> Viewer:
    _quiet_directwrite_font_warning()
    print(f"[DeepDeBlur3D] app {__version__} | torch {torch.__version__}")
    v = Viewer(title=f"deblur3d {__version__} — Inference")
    v.dims.ndisplay = 2
    v.grid.enabled = True

    state = {"run_idx": 1}
    relay = _Relay()
    panel = _ProgressPanel()
    abort_event = threading.Event()

    relay.progressed.connect(panel.set_progress)

    def _on_update(info: dict):
        commands = update_instructions(info.get("tag"))
        box = QMessageBox()
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Update available")
        box.setText(f"<b>DeepDeBlur3D {info['title']}</b> is available.")
        box.setInformativeText(
            f"You are running {__version__}.\n\n"
            "This does not install the update. Run:\n\n"
            f"{commands}"
        )
        box.setDetailedText(info.get("notes") or "The release has no notes.")
        open_button = box.addButton("Open release page", QMessageBox.ActionRole)
        copy_button = box.addButton("Copy update command", QMessageBox.ActionRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is open_button:
            import webbrowser
            webbrowser.open(info.get("url") or GITHUB_URL)
        elif clicked is copy_button:
            QApplication.clipboard().setText(commands)
            show_info("Update command copied to the clipboard.")
        panel.update_button.setEnabled(True)

    relay.update_found.connect(_on_update)

    def _check_for_update(announce_when_current: bool = False):
        # Network call, so it must not block startup or crash the app offline.
        try:
            info = app_update_available()
        except Exception:
            info = None
        if info:
            relay.update_found.emit(info)
        elif announce_when_current:
            relay.up_to_date.emit()

    def _on_up_to_date():
        panel.update_button.setEnabled(True)
        show_info(f"DeepDeBlur3D {__version__} is up to date.")

    relay.up_to_date.connect(_on_up_to_date)

    def _manual_update_check():
        panel.update_button.setEnabled(False)
        threading.Thread(
            target=lambda: _check_for_update(announce_when_current=True), daemon=True
        ).start()

    panel.update_button.clicked.connect(_manual_update_check)
    threading.Thread(target=_check_for_update, daemon=True).start()

    def _prepare_input_layer(layer: NapariImage) -> bool:
        if getattr(layer, "metadata", None) and layer.metadata.get("deblur3d_prepared"):
            return True
        data = np.asarray(layer.data)
        if data.ndim != 3:
            return False
        try:
            norm = normalize_float01(data)
        except Exception:
            return False
        layer.data = norm.astype(np.float32, copy=False)
        layer.colormap = "gray"
        _stabilize_contrast(layer, 0.0, 1.0)
        v.dims.ndisplay = 2
        v.grid.enabled = True
        return True

    def _update_run_enabled_from_active():
        active = v.layers.selection.active
        enable = isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) == 3
        infer_w.enabled = bool(enable) and not panel.abort_button.isEnabled()

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

    @magicgui(
        call_button="Run Filter",
        device={"choices": ["cuda", "cpu"]},
        preset={"label": "Tiling", "choices": list(TILE_PRESETS)},
        strength={"label": "Strength", "widget_type": "FloatSlider",
                  "min": 0.0, "max": 3.0, "step": 0.1},
        hp_sigma={"label": "HP Sigma (vox)", "widget_type": "FloatSlider",
                  "min": 0.0, "max": 8.0, "step": 0.1},
        hp_gain={"label": "HP Gain", "widget_type": "FloatSlider",
                 "min": 0.0, "max": 4.0, "step": 0.1},
        lp_gain={"label": "LP Gain", "widget_type": "FloatSlider",
                 "min": 0.0, "max": 4.0, "step": 0.1},
    )
    def infer_w(
        device: str = "cuda",
        preset: str = DEFAULT_PRESET,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
    ):
        def _is_cuda_error(err: Exception) -> bool:
            msg = str(err).lower()
            return ("cuda" in msg or "cudnn" in msg or "device-side assert" in msg)

        active = v.layers.selection.active
        if not (isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) == 3):
            show_warning(
                "Select a 3D image layer. For TIFF slices, use Napari's "
                "'Open Files as Stack' option."
            )
            return

        vol = normalize_float01(np.asarray(active.data))
        tile, overlap = TILE_PRESETS[preset]

        try:
            weights_path, config_path = ensure_model_assets(
                HFModelSpec(repo_id=HF_REPO_ID, weights_filename=HF_FILENAME),
                prompt=_ask_yes_no,
            )
        except Exception as e:
            show_error(f"Model resolution failed: {e}")
            return

        want_cuda = (device == "cuda")
        cuda_available = torch.cuda.is_available() if want_cuda else False
        first_device = "cuda" if (want_cuda and cuda_available) else "cpu"
        if want_cuda and not cuda_available:
            show_warning("CUDA not available. Falling back to CPU.")

        run_id = state["run_idx"]
        abort_event.clear()
        infer_w.enabled = False
        panel.abort_button.setEnabled(True)
        panel.reset("Starting…")

        def _fmt(x: float, nd=2):
            return f"{x:.1f}" if nd == 1 else f"{x:.2f}"

        def _launch(device_to_use: str):
            start = time.time()
            show_info(f"Starting inference on '{active.name}' using {device_to_use.upper()} …")
            bar = tqdm(total=1, desc=f"deblur3d #{run_id}", unit="tile")

            def on_progress(done: int, total: int):
                if bar.total != total:
                    bar.reset(total=total)
                bar.n = done
                bar.refresh()
                relay.progressed.emit(done, total)

            def _infer(v_, device=None, progress=None):
                # Aborting is a normal user action, so it returns None rather
                # than raising. An exception escaping the worker makes superqt
                # print a full traceback, which reads like a crash.
                try:
                    return run_inference(
                        v_,
                        device=device_to_use,
                        tile=tile,
                        overlap=overlap,
                        strength=strength,
                        hp_sigma=hp_sigma,
                        hp_gain=hp_gain,
                        lp_gain=lp_gain,
                        weights_path=weights_path,
                        config_path=config_path,
                        progress=on_progress,
                        should_abort=abort_event.is_set,
                    )
                except InferenceAborted:
                    return None

            worker = make_infer_worker(
                _infer, vol, device=device_to_use, extra_kwargs={},
            )

            def _finish():
                bar.close()
                panel.abort_button.setEnabled(False)
                infer_w.enabled = True

            def on_return(pred: Optional[np.ndarray]):
                dt = time.time() - start
                _finish()
                if pred is None:
                    panel.reset("Aborted")
                    show_info("Inference aborted.")
                    return
                panel.reset(f"Done in {dt:.1f}s")
                layer_name = (
                    f"filtered_s{_fmt(strength,1)}_"
                    f"hps{_fmt(hp_sigma,2)}_hpg{_fmt(hp_gain,2)}_lpg{_fmt(lp_gain,2)}_"
                    f"{run_id}"
                )
                lyr = v.add_image(
                    pred, name=layer_name, colormap="gray",
                    blending="translucent", opacity=0.7,
                    metadata={
                        "deblur3d_output": True,
                        "deblur3d": {
                            **provenance(config_path),
                            "device": device_to_use,
                            "preset": preset,
                            "tile": tile, "overlap": overlap,
                            "strength": strength, "hp_sigma": hp_sigma,
                            "hp_gain": hp_gain, "lp_gain": lp_gain,
                        },
                    },
                )
                _stabilize_contrast(lyr, 0.0, 1.0)
                lyr.grid_position = (0, 1)
                if active in v.layers:
                    v.layers.selection.active = active
                show_info(f"Inference #{run_id} done in {dt:.2f}s on {device_to_use.upper()} | shape={pred.shape}")
                v.grid.enabled = True
                state["run_idx"] = run_id + 1

            def on_error(e):
                print(f"[DeepDeBlur3D] ERROR on {device_to_use}: {e}")
                if device_to_use == "cuda" and _is_cuda_error(e):
                    bar.close()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    show_warning("CUDA run failed. Falling back to CPU automatically…")
                    _launch("cpu")
                else:
                    _finish()
                    panel.reset("Failed")
                    show_error(f"Inference error on {device_to_use}: {e}")

            worker.returned.connect(on_return)
            worker.errored.connect(on_error)
            worker.start()

        _launch(first_device)

    panel.abort_button.clicked.connect(lambda: (abort_event.set(), panel.label.setText("Aborting…")))

    from magicgui.widgets import Label

    heading = Label(value="Inference-time controls")
    heading.native.setStyleSheet("font-weight: bold; margin-top: 6px;")
    infer_w.insert(infer_w.index("strength"), heading)
    _make_readouts_editable(infer_w)
    _add_preset_info(infer_w)

    infer_w.enabled = False
    v.window.add_dock_widget(infer_w, name="DeepDeBlur3D", area="right")
    v.window.add_dock_widget(panel, name="Progress", area="right")
    return v


def main():
    build_viewer()
    run()


if __name__ == "__main__":
    main()
