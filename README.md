# 3DMM-Pipeline
End-to-end pipeline for building a 3D Morphable Model (3DMM) from raw 3D face meshes — including data preprocessing, symmetry augmentation, centering, rigid alignment, PCA model construction, reconstruction, and random face generation.

This repository provides a complete pipeline for constructing a **3D Morphable Model (3DMM)** from raw 3D face meshes. It covers every step of the classical 3DMM workflow — from dataset preparation and geometric alignment to PCA-based statistical modeling, reconstruction, and random face synthesis.  

It is designed to be **modular, reproducible, and research-ready**, making it useful for academic research, graphics pipelines, or as a foundation for downstream tasks such as 3DMM fitting, facial animation, and identity modeling.

---

## Architecture

```mermaid
flowchart TD
    A[Raw 3D Face Mesh Dataset]
    B[Symmetry Augmentation]
    B1[Mirror each mesh using predefined vertex correspondence]
    C[Centering and Normalization]
    D[Rigid Registration]
    D1[Procrustes alignment against reference / target shape]
    E[Offset Computation]
    E1[Compute vertex offsets relative to mean shape]
    F[Aligned Mean Shape]
    G[Incremental PCA]
    G1[Learn low-dimensional shape basis from offset data]
    H["Final 3DMM<br/>Shape Basis (PCA Components)<br/>Explained Variance<br/>Mean Shape / Mean Mesh"]

    A --> B
    B --> B1
    B1 --> C
    C --> D
    D --> D1
    D1 --> E
    D1 --> F
    E --> E1
    E1 --> G
    G --> G1
    G1 --> H
    F --> H

---

## Features

- **Data Preprocessing**
  - Automatic mirroring with symmetry index
  - Centering and normalization
  - Rigid alignment (Procrustes with optional scale)

- **3DMM Construction**
  - Offset computation relative to mean shape
  - Incremental PCA for large datasets
  - Explained variance and reconstruction error analysis

- **Model Usage**
  - Progressive reconstruction visualization
  - Random face generation from PCA parameter distribution
  - Reconstruction of unseen identities

- **Evaluation Tools**
  - RMSE reconstruction error plotting
  - PCA component analysis and visualization

---

### Required packages

- torch
- numpy
- scikit-learn
- matplotlib
- igl (Python bindings for libigl)
- torch.utils.data

### Installation

```bash
git clone https://github.com/<your-username>/3DMM-Construction-Pipeline.git
cd 3DMM-Construction-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt




