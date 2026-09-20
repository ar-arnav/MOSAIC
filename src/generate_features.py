import os
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

# Import the tool we just defined
from features import format_spatial_features 
from vectorizer import RTFVectorizer
# We NO LONGER need to import SpatialPipeline!

# ---------------- CONFIGURATION ---------------- #
DATA_DIR = Path("data/processed/full_pipeline")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLADE_TREE_PATH = Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl").expanduser()
GLADE_HEALPIX_PATH = Path("~/MOSAIC/data/processed/GLADE+/glade_healpix_nside128.pkl").expanduser()
RTF_ONNX_PATH = Path("models/rtf_encoder_latents.onnx").expanduser() 
RTF_STATS_PATH = Path("docs/Model W&B/RTF_MOSAIC/mosaic_rtf_stats.json").expanduser()
# GW_FITS_PATH and GW_T0 are no longer needed!

def load_onnx_model(path):
    if not path.exists():
        raise FileNotFoundError(f"RTF ONNX model not found at {path}")
    return ort.InferenceSession(str(path))

def process_split(split_name):
    print(f"\n{'='*60}")
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print(f"{'='*60}")

    parquet_path = DATA_DIR / f"{split_name}.parquet"
    if not parquet_path.exists():
        print(f"Skipping {split_name}: File not found.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} alerts from {parquet_path}")

    # 1. Initialize RTF Vectorizer
    print("Initializing Vectorizer...")
    vectorizer = RTFVectorizer(stats_path=str(RTF_STATS_PATH))
    
    # 2. Load RTF ONNX Model
    print("Loading RTF ONNX Model...")
    ort_session = load_onnx_model(RTF_ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    # We assume the mask input is 'pad_mask' based on previous errors
    mask_input_name = 'pad_mask'

    unique_ids = df['objectId'].unique()
    N = len(unique_ids)
    print(f"Found {N} unique transients.")

    all_rtf_latents = []
    all_spatial_feats = []
    all_labels = []

    for i, obj_id in enumerate(unique_ids):
        if i % 100 == 0:
            print(f" Progress: {i}/{N}", end='\r')

        sub_df = df[df['objectId'] == obj_id].sort_values('jd')
        
        # --- LABELS ---
        # Use 'label_class' from your columns
        label = sub_df['label_class'].iloc[0]

        try:
            # --- STREAM 1: RTF Latents ---
            rtf_matrix, mask = vectorizer.vectorize(sub_df)
            
            ort_inputs = {
                input_name: rtf_matrix.astype(np.float32),
                mask_input_name: mask.astype(bool)
            }
            rtf_latent = ort_session.run(None, ort_inputs)[0].flatten() 

            # --- STREAM 2: Spatial Features (EXTRACT FROM COLUMNS) ---
            # We manually build the dictionary that format_spatial_features expects
            # using the columns from your parquet file.
            
            # Calculate is_coinc based on logic (cred <= 0.9 and dp_dv > 0)
            cred = sub_df['gw_cred_level'].iloc[0]
            dp_dv = sub_df['gw_dp_dv'].iloc[0]
            is_coinc_bool = (cred <= 0.9) and (dp_dv > 0.0)

            spatial_dict = {
                "has_gw_coincidence": is_coinc_bool,
                "gw_cred_level": cred,
                "dp_dv": dp_dv,
                "best_host_dist_mpc": sub_df['dist_mpc'].iloc[0],
                "best_host_K_mag": sub_df['host_K_mag'].iloc[0],
                "S_gal": sub_df['host_S_gal'].iloc[0]
            }

            # Use the formatting function
            spatial_feat = format_spatial_features(spatial_dict)

            all_rtf_latents.append(rtf_latent)
            all_spatial_feats.append(spatial_feat)
            all_labels.append(label)

        except Exception as e:
            print(f"\nError processing {obj_id}: {e}")
            all_rtf_latents.append(np.zeros(128, dtype=np.float32))
            all_spatial_feats.append(np.zeros(6, dtype=np.float32))
            all_labels.append(label)

    # Save Arrays
    print(f"\nSaving {split_name} features...")
    np.save(OUTPUT_DIR / f"{split_name}_rtf_latents.npy", np.array(all_rtf_latents))
    np.save(OUTPUT_DIR / f"{split_name}_spatial_features.npy", np.array(all_spatial_feats))
    np.save(OUTPUT_DIR / f"{split_name}_ground_truth.npy", np.array(all_labels).reshape(-1, 1))
    
    print(f"Finished {split_name}. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        process_split(split)