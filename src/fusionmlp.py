import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    @classmethod
    def from_pretrained(cls, weights_path, device='cpu'):
        model = cls()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model

    def predict(self, rtf_score, gw_cred_level, gw_dp_dv):
        features = torch.tensor([[rtf_score, gw_cred_level, gw_dp_dv]], dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(self(features).squeeze(-1)).item()
        return prob

    def predict_batch(self, features_array):
        features = torch.tensor(features_array, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self(features).squeeze(-1)).numpy()
        return probs