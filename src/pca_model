import os
import numpy as np
from sklearn.decomposition import IncrementalPCA

class IncrementalPCAIterator:
    """Fit incremental PCA on large vertex datasets."""

    def __init__(self, dataloader, n_components=100):
        self.dataloader = dataloader
        self.pca = IncrementalPCA(n_components=n_components, batch_size=dataloader.batch_size)
        self.fitted = False
    
    def run(self):
        for batch in self.dataloader:
            verts = batch['vertices'].numpy()
            flat = verts.reshape(verts.shape[0], -1)
            self.pca.partial_fit(flat)
        self.fitted = True
        return self.pca

def save_incremental_pca(pca_model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'pca_components.npy'), pca_model.components_)
    np.save(os.path.join(save_dir, 'pca_mean.npy'), pca_model.mean_)
    np.save(os.path.join(save_dir, 'pca_explained_variance.npy'), pca_model.explained_variance_)
    np.save(os.path.join(save_dir, 'pca_explained_variance_ratio.npy'), pca_model.explained_variance_ratio_)
