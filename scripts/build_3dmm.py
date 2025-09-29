"""
Example pipeline script to build a 3DMM from raw meshes.
Run with: python scripts/build_3dmm.py
"""

from src.dataset import CustomMeshDataset, NumpyVertexDataset
from src.transforms import MirrorTransform, Centering, RigidAligner
from src.preprocessing import VertsExportor, MeanIterator, RegistrationIterator, OffsetCreator
from src.pca_model import IncrementalPCAIterator, save_incremental_pca
from torch.utils.data import DataLoader
import numpy as np

# Mirror export
dataset = CustomMeshDataset("meshes/")
mirror = MirrorTransform(symmetry_idx=np.load("symmetry_idx.npy"))
VertsExportor(dataset, "verts_npy/", transform=mirror).run()

# Center and compute mean
center_ds = NumpyVertexDataset("verts_npy/")
center_loader = DataLoader(center_ds, batch_size=1)
mean_it = MeanIterator(center_loader, save_dir="centered_npy/", transform=Centering())
mean_shape = mean_it.compute_mean_dataset().numpy()
np.save("aligned_mean.npy", mean_shape)

# Register
aligner = RigidAligner(target=torch.tensor(mean_shape))
centered_loader = DataLoader(NumpyVertexDataset("centered_npy/"), batch_size=1)
reg_it = RegistrationIterator(centered_loader, aligner, save_dir="processed_npy/")
aligned_mean = reg_it.run().numpy()
np.save("aligned_mean.npy", aligned_mean)

# Offsets + PCA
files = [f for f in os.listdir("processed_npy/") if f.endswith("_processed.npy")]
OffsetCreator("processed_npy/", "aligned_mean.npy").offset(files, save_dir="offsets/")
train_loader = DataLoader(NumpyVertexDataset("offsets/"), batch_size=32, shuffle=True)
pca_model = IncrementalPCAIterator(train_loader, n_components=100).run()
save_incremental_pca(pca_model, "pca/")
