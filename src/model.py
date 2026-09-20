import torch
import torch.nn as nn

class MOSAICFusionMLP(nn.Module):
    """
    Stream 3: Multi-Stream Fusion MLP for MOSAIC.
    Concatenates Stream 1 (128-dim RTF Autoencoder latent vector) and 
    Stream 2 (6-dim Spatial/Astrophysical feature vector) into a 134-dim input space.
    """
    def __init__(self, rtf_dim=128, spatial_dim=6, dropout_rate=0.3):
        super(MOSAICFusionMLP, self).__init__()
        
        input_dim = rtf_dim + spatial_dim  # 134 features total
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
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
        """
        rtf_vector: Tensor of shape [batch_size, 128]
        spatial_vector: Tensor of shape [batch_size, 6]
        Returns: P(KN) tensor of shape [batch_size, 1]
        """
        fused_vector = torch.cat((rtf_vector, spatial_vector), dim=1)
        return self.network(fused_vector)