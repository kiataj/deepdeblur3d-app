DeepDeblur3D GUI

Napari-based GUI for DeepDeblur3D with automatic model download from Hugging Face.

Quick Start
Prereqs

Python 3.10 (Conda recommended)

Git

(Optional GPU) NVIDIA driver supporting CUDA 11.6+ (≥ 511.xx on Windows).
If unsure: nvidia-smi shows your driver.

1) Clone
git clone https://github.com/<YOUR-ORG>/DeepDeBlur3D.git
cd DeepDeBlur3D

2) Create env (recommended)
# Windows / Linux / macOS (Conda)
conda create -n deblur3d python=3.10 -y
conda activate deblur3d

3) Base install (GUI + deps)
pip install -e .

4) Choose PyTorch backend
A) CPU (portable)
pip install -e .[cpu]


This installs CPU wheels (torch==1.12.1, torchvision==0.13.1).

B) GPU (CUDA 11.6 wheels)

Update your NVIDIA driver first (≥ 511.xx), then:

pip install -e .[cu116] --extra-index-url https://download.pytorch.org/whl/cu116


This installs torch==1.12.1+cu116 and matching torchvision.

Note: CUDA wheels come from PyTorch’s extra index; that’s why --extra-index-url is required.

5) Sanity check
python - << "PY"
import torch, napari, numpy
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Napari:", napari.__version__)
print("NumPy:", numpy.__version__)
PY

6) Launch
deblur3d-gui