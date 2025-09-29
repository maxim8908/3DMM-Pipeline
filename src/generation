import os
import numpy as np
import igl

class PCAFaceGenerator:
    """Reconstruct a face mesh from PCA components and offsets."""

    def __init__(self, pca_components_path, mean_path, face_path):
        self.components = np.load(pca_components_path)
        self.mean = np.load(mean_path).reshape(-1)
        self.face = np.load(face_path)
        self.num = self.mean.shape[0] // 3

    def meshes(self, offset_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        offset = np.load(offset_path).reshape(-1)
        p = self.components @ offset
        recon = self.mean + self.components.T @ p
        verts = recon.reshape(self.num, 3)
        mesh_path = os.path.join(output_dir, "reconstructed.obj")
        igl.writeOBJ(mesh_path, verts, self.face)
        np.save(os.path.join(output_dir, 'params.npy'), p)

class PCARandomFaceGenerator:
    """Generate new random face meshes from learned PCA distribution."""

    def __init__(self, pca_dir, face_dir, mean_path):
        self.mean = np.load(mean_path).reshape(-1)
        self.components = np.load(os.path.join(pca_dir, 'pca_components.npy'))
        self.faces = np.load(os.path.join(face_dir, 'face.npy'))
        self.num = self.mean.shape[0] // 3
    
    def sample_parameters(self, mu, cov):
        return np.random.multivariate_normal(mu, cov)
    
    def reconstruction(self, alpha):
        return (self.mean + self.components.T @ alpha).reshape(self.num, 3)
    
    def generate_save(self, save_path, mu, cov):
        alpha = self.sample_parameters(mu, cov)
        vertices = self.reconstruction(alpha)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        igl.writeOBJ(save_path, vertices, self.faces)
        return vertices, alpha
