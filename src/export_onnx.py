import sys
import os
import torch
import torch.nn as nn
import torch.onnx # Changed from torch.export to standard onnx for compatibility
import json
import numpy as np

# Point directly to the inner rtf/src directory where model.py actually lives
current_dir = os.path.dirname(os.path.abspath(__file__))
rtf_src_path = os.path.abspath(os.path.join(current_dir, "../rtf/src"))
if rtf_src_path not in sys.path:
    sys.path.insert(0, rtf_src_path)

# type: ignore[import-not-found]
from model import LightCurveCompressor

IN_CHANNELS = 18
LATENT_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CHANGED CLASS NAME & LOGIC ---
# We only want the Encoder, not the Classifier
class RTFEncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the LightCurveCompressor
        self.rtf = LightCurveCompressor(
            mode="ae", latent_dim=LATENT_DIM, in_channels=IN_CHANNELS,
            d_model=256, use_images=False, gp_dim=0, num_classes=0,
        )
        # REMOVED self.head - we don't want the classification score

    def forward(self, x, pad_mask):
        # Return the EMBEDDING (128 features), not the classification
        return self.rtf.embed(x, pad_mask)
# --------------------------------

def export_onnx(checkpoint_path="docs/Model W&B/RTF_MOSAIC/rtf_encoder_clf_calibrated.pt", 
                filename="models/rtf_encoder_latents.onnx",  
                stats_filename="models/RTF_MOSAIC/mosaic_rtf_stats.json"):
    
    model = RTFEncoderOnly().to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    # The checkpoint might contain 'rtf.*' and 'head.*'. 
    # We only load 'rtf.*' parts or strict=False to ignore the head.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Extract and save global stats for the runtime vectorizer
    if "stats" in ckpt:
        with open(stats_filename, "w") as f:
            json.dump(ckpt["stats"], f, indent=4)
        print(f"Exported normalization stats to {stats_filename}")
    else:
        print("WARNING: No 'stats' key found in the checkpoint file.")

    # Dummy inputs for tracing (Batch=1, SeqLen=257, Channels=18) -- Increased to match vectorizer
    batch_size, seq_len, channels = 1, 257, IN_CHANNELS
    dummy_input = torch.randn(batch_size, seq_len, channels, device=device)
    dummy_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    # Standard dynamic axes for torch.onnx.export
    dynamic_axes = {
        'input_features': {0: 'batch_size', 1: 'seq_len'},
        'pad_mask': {0: 'batch_size', 1: 'seq_len'},
        'latent_vector': {0: 'batch_size'} # <--- CHANGED OUTPUT NAME
    }

    print(f"Exporting ONNX model to {filename}...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_pad_mask),
        filename,
        opset_version=17, # Slightly older opset is more stable generally
        input_names=["input_features", "pad_mask"],
        output_names=["latent_vector"], # <--- CHANGED OUTPUT NAME
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX model successfully exported to {filename}")

if __name__ == "__main__":
    export_onnx()