import os
import torch
import numpy as np

class VertsExportor:
    """Save original and mirrored vertices to .npy."""

    def __init__(self, dataset, output_dir, transform=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.transform = transform
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self):
        for sample in self.dataset:
            name = sample['name'].replace('.obj', '')
            verts = sample['vertices']
            np.save(os.path.join(self.output_dir, f"{name}.npy"), verts.numpy())
            if self.transform:
                mirrored = self.transform(verts)
                np.save(os.path.join(self.output_dir, f"{name}_mirrored.npy"), mirrored.numpy())

class MeanIterator:
    """Compute and save centered meshes and return global mean."""

    def __init__(self, dataloader, save_dir, transform=None):
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.transform = transform
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_mean_dataset(self):
        sum_vertices, total_count = None, 0
        for batch in self.dataloader:
            verts = batch['vertices'][0]
            name = batch['name'][0].replace('.npy', '')
            centered = self.transform(verts) if self.transform else verts
            np.save(os.path.join(self.save_dir, f"{name}_centered.npy"), centered.numpy())
            sum_vertices = centered.clone() if sum_vertices is None else sum_vertices + centered
            total_count += 1
        return sum_vertices / total_count

class RegistrationIterator:
    """Rigidly register all meshes to a target and compute aligned mean."""

    def __init__(self, dataloader, aligner, save_dir):
        self.dataloader = dataloader
        self.aligner = aligner
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def run(self):
        sum_vertices, total_count = None, 0
        for batch in self.dataloader:
            verts = batch['vertices'][0]
            name = batch['name'][0].replace('_centered.npy', '')
            registered = self.aligner(verts)
            np.save(os.path.join(self.save_dir, f"{name}_processed.npy"), registered.numpy())
            sum_vertices = registered.clone() if sum_vertices is None else sum_vertices + registered
            total_count += 1
        return sum_vertices / total_count

class IdentitySplitter:
    """Split dataset identities into train/val sets."""

    def __init__(self, id_dir, val_count=10, seed=42):
        self.id_dir = id_dir
        self.val_count = val_count
        self.seed = seed
    
    def get_identities(self):
        files = os.listdir(self.id_dir)
        ids = set(f.replace('_mirrored', '').replace('_processed.npy', '') for f in files if f.endswith('_processed.npy'))
        return sorted(ids)
    
    def split(self):
        import random
        ids = self.get_identities()
        random.seed(self.seed)
        random.shuffle(ids)
        return ids[self.val_count:], ids[:self.val_count]
    
    def split_file(self, identities):
        all_files = os.listdir(self.id_dir)
        return [f for f in all_files if f.endswith('_processed.npy') and any(base in f for base in identities)]

class OffsetCreator:
    """Compute vertex offset relative to mean shape."""

    def __init__(self, id_dir, mean_shape_path):
        self.id_dir = id_dir
        self.mean = np.load(mean_shape_path)

    def offset(self, file_list, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for fname in file_list:
            verts = np.load(os.path.join(self.id_dir, fname))
            offset = verts - self.mean
            np.save(os.path.join(save_dir, f"{fname.replace('_processed.npy', '')}_offset.npy"), offset)
