import torch
import torch.nn as nn

class MOSAICOpticalOnly(nn.Module):
    """
    Ablation Study Model: Uses ONLY Light Curves (RTF).
    Intentionally ignores Spatial/GW features to test optical performance.
    """
    def __init__(self, rtf_dim=128, dropout_rate=0.3):
        super(MOSAICOpticalOnly, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(rtf_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, rtf_vector, spatial_vector):
        # spatial_vector is passed in to match dataset signature, but ignored.
        return self.network(rtf_vector)