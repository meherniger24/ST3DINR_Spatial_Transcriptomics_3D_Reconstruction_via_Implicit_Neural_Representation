# ST3D-INR

### 3D Spatial Transcriptomics Reconstruction via Implicit Neural Representation

A computational pipeline for reconstructing continuous 3D tissue architecture from sparse serial H&E sections with spatial transcriptomics annotations, using implicit neural representations (INR) to interpolate gene expression between slices.

---

## Overview

Most spatial transcriptomics datasets are acquired as 2D sections. This pipeline takes a small number of annotated serial sections (as few as 3) and reconstructs a full 3D tissue volume by learning a continuous mathematical function that maps any 3D spatial coordinate to gene expression — then querying that function at positions where no real tissue slice exists.

This extends the [STINR](https://github.com/YisiLuo/STINR) framework (Luo et al., CVPR 2025) — originally designed for within-slice deconvolution and denoising — to perform **between-slice interpolation for 3D reconstruction**, a use case not explored in the original paper.

---

## How It Works

STINR trains a **SIREN (Sinusoidal Implicit Neural Representation)** that learns a continuous function:

```
(x, y, z) → gene expression → cell-type proportions
```

While the original paper only evaluates the model at observed slice positions, the learned function exists everywhere in 3D space. We exploit this by querying the model at z-values between real slices, generating biologically informed virtual sections — then converting those into 3D meshes.

---

## Pipeline

```
Serial H&E Sections (3 or 9 slices)
            │
            ▼
  01_data_preparation.py
  - Normalize gene expression
  - Build spatial neighbor graphs per slice
  - Derive cell-type basis matrix from annotations
  - Output STINR-compatible h5ad files
            │
            ▼
  02_train_stinr.py
  - Train STINR on all slices jointly
  - SIREN encoder maps (x,y,z) → latent space
  - Deconvolution decoder maps latent → cell-type proportions
  - Save trained model weights
            │
            ▼
  03_interpolate_slices.py
  - Query trained INR at intermediate z-values (every 5µm)
  - Generate virtual slices with predicted cell-type proportions
  - Output per-slice h5ad files with 3D coordinates
            │
            ▼
  04_build_meshes.py
  - Build 3D density volumes per cell type
  - Apply Gaussian smoothing
  - Extract isosurfaces via marching cubes
  - Export OBJ meshes ready for Blender
```

---

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/ST3D-INR.git
cd ST3D-INR

# Clone STINR dependency
git clone https://github.com/YisiLuo/STINR.git

# Install Python dependencies
pip install -r requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU recommended for training (tested on NVIDIA A40)
- Apple MPS supported for inference

```
torch>=2.0.0
scanpy>=1.9.0
anndata>=0.9.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
scipy>=1.9.0
numpy>=1.23.0
pandas>=1.5.0
tqdm
```

---

## Data Format

Input: a single `.h5ad` file containing all slices with the following fields:

| Field | Description |
|---|---|
| `adata.X` | Raw gene expression counts (sparse matrix) |
| `adata.obs['sample']` | Slice name (e.g., `'B01'`, `'E01'`) |
| `adata.obs['anno_initial']` | Cell-type annotations |
| `adata.obs['z']` | Physical z-position in µm |
| `adata.obsm['spatial']` | 3D coordinates (x, y, z) |

---

## Usage

### Step 1 — Data Preparation

```python
# Edit these paths at the top of the script
INPUT_PATH = "/path/to/combined_annotations.h5ad"
OUTPUT_DIR = "/path/to/stinr_data"
SLICES     = ['B01', 'E01', 'E02']   # or all 9 slices
```

```bash
python 01_data_preparation.py
```

---

### Step 2 — Train STINR

```python
# Edit these settings at the top of the script
DATA_DIR             = "/path/to/stinr_data"
STINR_PATH           = "/path/to/STINR"
SAVE_DIR             = "/path/to/stinr_results"
SUBSAMPLE_PER_SLICE  = None    # None = full data, or e.g. 20000 for testing
TRAINING_STEPS       = 14001
```

```bash
# Local
python 02_train_stinr.py

# HPC cluster (LSF)
bsub < lsf_scripts/submit_train.lsf
```

---

### Step 3 — Interpolate Virtual Slices

```python
# Edit these settings at the top of the script
DATA_DIR    = "/path/to/stinr_data"
RESULTS_DIR = "/path/to/stinr_results"
OUTPUT_DIR  = "/path/to/interpolated_slices"
STINR_PATH  = "/path/to/STINR"
REAL_SLICES = {'B01': 0.0, 'E01': 30.0, 'E02': 80.0}   # slice name → z in µm
Z_STEP      = 5.0    # µm between virtual slices
```

```bash
python 03_interpolate_slices.py
```

---

### Step 4 — Build 3D Meshes

```python
# Edit these settings at the top of the script
INTERPOLATED_DIR = "/path/to/interpolated_slices"
OUTPUT_DIR       = "/path/to/meshes_3d"
GRID_RESOLUTION  = 300
N_Z_INTERP       = 300
CELL_SIGMA_XY    = 4.0
CELL_SIGMA_Z     = 10.0
LEVEL_FRAC       = 0.10
```

```bash
python 04_build_meshes.py
```

Output: one `.obj` mesh file per cell type, ready to import into Blender.

---

## Citation

If you use this pipeline, please cite the original STINR paper:

```bibtex
@inproceedings{luo2025stinr,
  title     = {STINR: Deciphering Spatial Transcriptomics via Implicit Neural Representation},
  author    = {Luo, Yisi and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

---

## Acknowledgements

This pipeline builds on [STINR](https://github.com/YisiLuo/STINR) (Luo et al., CVPR 2025). The 3D interpolation and mesh reconstruction components are novel extensions of the original framework.

---

## License

MIT License
