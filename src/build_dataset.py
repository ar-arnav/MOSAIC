import os
import numpy as np
import pandas as pd
import torch

def compile_split(split_name):
    print(f"\n---------------------------------------------------------")
    print(f"Processing Split: [{split_name.upper()}]")
    print(f"---------------------------------------------------------")
    
    # 1. Read from data/processed/full_pipeline/
    pipeline_dir = os.path.join("data", "processed", "full_pipeline")
    parquet_path = os.path.join(pipeline_dir, f"{split_name}.parquet")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Critical: Parquet file for '{split_name}' not found at '{parquet_path}'. "
            f"Please verify that '{split_name}.parquet' exists in '{pipeline_dir}'."
        )
        
    print(f"Reading {split_name} parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Count unique transient candidate objects
    N = len(df['objectId'].unique()) if 'objectId' in df.columns else len(df)
    print(f"Loaded {split_name} dataframe. Unique sources (N): {N}")
    
    # 2. Load Corresponding Feature Arrays (Stream 1 & Stream 2)
    # Looking in outputs/ for split-specific feature arrays
    rtf_path = os.path.join("models/outputs", f"{split_name}_rtf_latents.npy")
    spatial_path = os.path.join("models/outputs", f"{split_name}_spatial_features.npy")
    labels_path = os.path.join("outputs", f"{split_name}_ground_truth.npy")
    
    for path in [rtf_path, spatial_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required feature file: '{path}'. Ensure your extraction scripts output split-specific arrays.")
            
    rtf_latents = np.load(rtf_path).astype(np.float32)
    spatial_features = np.load(spatial_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1) # Ensure [N, 1] shape
        
    # 3. Dimensionality Assertions
    assert rtf_latents.shape[0] == spatial_features.shape[0] == labels.shape[0] == N, \
        f"Dimension mismatch in {split_name} split across features and source count!"
    assert rtf_latents.shape[1] == 128, f"Expected RTF dimension 128, got {rtf_latents.shape[1]}"
    assert spatial_features.shape[1] == 6, f"Expected Spatial dimension 6, got {spatial_features.shape[1]}"

    # 4. Save Compressed .npz Archive directly into data/processed/full_pipeline/
    os.makedirs(pipeline_dir, exist_ok=True)
    out_file_path = os.path.join(pipeline_dir, f"{split_name}.npz")
    
    np.savez_compressed(
        out_file_path,
        rtf=rtf_latents,
        spatial=spatial_features,
        label=labels  # Matches dataset.py key expectation
    )
    
    pos_count = int(labels.sum())
    neg_count = N - pos_count
    print(f"-> Successfully compiled {out_file_path} | Samples: {N} | Kilonovas: {pos_count} | Background: {neg_count}")

def main():
    print("=" * 65)
    print("MOSAIC PIPELINE: Full Pipeline Parquet Compilation Engine")
    print("=" * 65)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        compile_split(split)
        
    print("\n" + "=" * 65)
    print("All split archives successfully compiled inside data/processed/full_pipeline/")
    print("=" * 65)

if __name__ == "__main__":
    main()