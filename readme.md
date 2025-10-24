# DeepDeblur3D GUI

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-orange)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-lightgrey)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Napari-based GUI for DeepDeblur3D** with automatic model download from Hugging Face.
---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kiataj/deepdeblur3d-app.git
cd deepdeblur3d-app
```

### 2️⃣ Create conda environment (recommended)
##### Windows / Linux / macOS (Conda)

```
conda create -n deblur3d python=3.10 -y
conda activate deblur3d
```

### 3️⃣ Base install (GUI + deps)
```bash
pip install -e .
```

### 4️⃣ Choose PyTorch backend
#### A) CPU (portable)
```
pip install -e .[cpu]
```

This installs CPU wheels (torch==1.12.1, torchvision==0.13.1).

#### B) GPU (CUDA 11.6 wheels)

Update your NVIDIA driver first (≥ 511.xx), then:
```
pip install -e .[cu116] --extra-index-url https://download.pytorch.org/whl/cu116
```

This installs torch==1.12.1+cu116 and matching torchvision.


### 5️⃣ Launch
```
deblur3d-gui
```
