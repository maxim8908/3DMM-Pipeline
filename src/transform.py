import torch
import numpy as np

class MirrorTransform:
    """Mirror mesh vertices across the symmetry axis."""

    def __init__(self, symmetry_idx):
        self.symmetry = torch.as_tensor(symmetry_idx, dtype=torch.long)
    
    def __call__(self, vertices: torch.Tensor) -> torch.Tensor:
        v = vertices[self.symmetry].clone()
        v[:, 0] = -v[:, 0]
        return v

class Centering:
    """Center mesh vertices around the origin."""

    def __call__(self, vertices: torch.Tensor) -> torch.Tensor:
        center = vertices.mean(dim=0, keepdim=True)
        return vertices - center

class RigidAligner:
    """Align a source mesh to a target using SVD (rotation + optional scale)."""

    def __init__(self, target, with_scale=False):
        self.target = target
        self.with_scale = with_scale
    
    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        X = source - source.mean(dim=0, keepdim=True)
        Y = self.target - self.target.mean(dim=0, keepdim=True)
        H = X.T @ Y
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        scale = S.sum() / (X.pow(2).sum()) if self.with_scale else 1.0
        t = self.target.mean(dim=0, keepdim=True) - scale * (source.mean(dim=0, keepdim=True) @ R)
        return scale * (source @ R) + t
