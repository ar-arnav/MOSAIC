import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MOSAICDataset(Dataset):
    """
    PyTorch Dataset for MOSAIC Stream 3 Fusion Network.
    Optimized for real multi-messenger astrophysical streams (ZTF + LIGO/GLADE+).
    Includes automatic NaN/Inf sanitization for real-world catalog stability.
    """
    def __init__(self, split='train', data_dir='data/processed/full_pipeline/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        
        file_path = os.path.join(self.data_dir, f"{self.split}.npz")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file for split '{self.split}' not found at path: {file_path}.\n"
                f"Ensure your full pipeline compilation script ran and generated archives in: {self.data_dir}"
            )
        
        # Load compressed archive from disk
        archive = np.load(file_path)
        
        rtf_arr = archive['rtf']         # Expected shape: [N, 128]
        spatial_arr = archive['spatial']   # Expected shape: [N, 6]
        
        # Support both 'labels' and 'label' key naming conventions
        if 'labels' in archive:
            labels_arr = archive['labels']
        elif 'label' in archive:
            labels_arr = archive['label']
        else:
            raise KeyError(f"Archive keys found: {archive.files}. Expected 'label' or 'labels'.")
        
        # ---------------------------------------------------------
        # REAL-DATA SANITIZATION
        # ---------------------------------------------------------
        # Real catalog cross-matches (GLADE+) or light-curve feature extractions 
        # can occasionally produce NaNs or infs. This replaces them safely to 
        # prevent catastrophic loss divergence during backpropagation.
        rtf_arr = np.nan_to_num(rtf_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        spatial_arr = np.nan_to_num(spatial_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Convert to floating-point PyTorch tensors for CPU/GPU compute
        self.rtf_features = torch.tensor(rtf_arr, dtype=torch.float32)
        self.spatial_features = torch.tensor(spatial_arr, dtype=torch.float32)
        self.labels = torch.tensor(labels_arr, dtype=torch.float32)
        
        # Ensure correct label dimensions: [N, 1]
        if self.labels.ndim == 1:
            self.labels = self.labels.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'rtf': self.rtf_features[idx],
            'spatial': self.spatial_features[idx],
            'label': self.labels[idx]
        }