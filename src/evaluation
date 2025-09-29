import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

class ReconstructionErrors:
    """Compute PCA reconstruction error for different component counts."""

    def __init__(self, pca_components_path, dataloader):
        self.components = np.load(pca_components_path)
        self.dataloader = dataloader
        self.errors = {}
    
    def evaluation(self, component_sizes):
        self.errors.clear()
        for k in component_sizes:
            B = self.components[:k]
            total_error, count = 0, 0
            for batch in self.dataloader:
                verts = batch['vertices'].numpy().reshape(-1)
                p = B @ verts
                recon = B.T @ p
                error = root_mean_squared_error(verts, recon)
                total_error += error
                count += 1
            self.errors[k] = total_error / count
    
    def plot_error(self, title):
        ks, es = list(self.errors.keys()), list(self.errors.values())
        plt.figure(figsize=(8, 5))
        plt.plot(ks, es, marker='o')
        plt.title(title)
        plt.xlabel("Number of Components")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
