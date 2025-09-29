import os
import torch
import numpy as np
import igl
from torch.utils.data import Dataset

class CustomMeshDataset(Dataset):
    """Load raw .obj meshes as vertex tensors."""

    def __init__(self, mesh_dir, transform=None):
        self.mesh_dir = mesh_dir
        self.meshfile_names = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")]
        self.transform = transform
    
    def __len__(self):
        return len(self.meshfile_names)
    
    def __getitem__(self, idx):
        meshfile_name = self.meshfile_names[idx]
        meshfile_path = os.path.join(self.mesh_dir, meshfile_name)
        v, _ = igl.read_triangle_mesh(meshfile_path)
        vertices = torch.tensor(v, dtype=torch.float32)
        if self.transform:
            vertices = self.transform(vertices)
        return {"vertices": vertices, "name": meshfile_name}

class NumpyVertexDataset(Dataset):
    """Load pre-processed vertices saved as .npy files."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        vertices = torch.tensor(np.load(file_path), dtype=torch.float32)
        if self.transform:
            vertices = self.transform(vertices)
        return {"vertices": vertices, "name": file_name}
