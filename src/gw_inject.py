import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import astropy_healpix as ah
from gw_processor import GW_data

gw_dir = os.path.expanduser("~/MOSAIC/data/raw/GW/")
all_files = glob.glob(os.path.join(gw_dir, "*.fits"))

np.random.seed(42)
all_files = np.array(all_files)
np.random.shuffle(all_files)

num_files = len(all_files)
train_split = int(0.7 * num_files)
val_split = int(0.85 * num_files)

train_maps = all_files[:train_split]
val_maps = all_files[train_split:val_split]
test_maps = all_files[val_split:]

print(f"Map split: {len(train_maps)} Train, {len(val_maps)} Val, {len(test_maps)} Test")

def generate_pairs(optical_path, gw_maps, out_path):
    optical_df = pd.read_parquet(optical_path)
    
    kn_ids = optical_df[optical_df['label_class'] == 1]['objectId'].unique()
    snibc_ids = optical_df[(optical_df['label_class'] == 0) & (optical_df['subclass'] == 'SNIbc')]['objectId'].unique()
    other_imp_ids = optical_df[(optical_df['label_class'] == 0) & (optical_df['subclass'] != 'SNIbc')]['objectId'].unique()
    
    min_kn = 25
    min_snibc = 10
    total_size = 200
    
    results = []
    for map_path in gw_maps:
        kn_sample = np.random.choice(kn_ids, size=min(min_kn, len(kn_ids)), replace=False)
        snibc_sample = np.random.choice(snibc_ids, size=min(min_snibc, len(snibc_ids)), replace=False)
        
        remaining = total_size - len(kn_sample) - len(snibc_sample)
        other_sample = np.random.choice(other_imp_ids, size=min(remaining, len(other_imp_ids)), replace=False)
        
        batch = np.concatenate([kn_sample, snibc_sample, other_sample])
        np.random.shuffle(batch)
        
        gw = GW_data(map_path)
        print(f'Processed {map_path}...')

        for i in batch:
            row = optical_df[optical_df['objectId'] == i].iloc[0]
            label = row['label_class']
            
            if label == 1:
                # KILONOVA: True GW Counterpart
                # 1. Sample sky pixel proportional to GW 2D spatial probability density
                pix_idx = np.random.choice(len(gw.pixel_prob), p=gw.pixel_prob)
                uniq_val = gw.uniq[pix_idx]
                lvl, px = ah.uniq_to_level_ipix(uniq_val)
                ns = ah.level_to_nside(lvl)
                lon, lat = ah.healpix_to_lonlat(px, ns, order='nested')
                
                ra = np.degrees(lon).value + np.random.uniform(-0.05, 0.05)
                dec = np.degrees(lat).value + np.random.uniform(-0.05, 0.05)
                
                # 2. Distance MUST match the 3D GW posterior at this pixel (mu, sigma)
                # Drawing distance from the map's pixel posterior guarantees physical consistency
                dist = np.random.normal(gw.distmu[pix_idx], gw.distsig[pix_idx])
                dist = max(5.0, dist)  # Enforce positive physical distance
                
            else:
                # IMPOSTER: Uncorrelated Background Alert
                # 1. Position is uniform across the sky (uncorrelated with GW beam)
                ra = np.random.uniform(0, 360)
                dec = np.degrees(np.arcsin(np.random.uniform(-1, 1)))
                
                # 2. Distance comes from independent 10-200 Mpc volumetric sampling
                dist = row['dist_mpc']

            cred_level, dp_dv = gw.evaluate_candidate(ra, dec, dist)
            results.append({
                'object_id': i, 
                'label': label, 
                'gw_cred_level': 1 - cred_level, 
                'gw_dp_dv': dp_dv
            })

    df_results = pd.DataFrame(results)
    df_results.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    generate_pairs(
        optical_path=os.path.expanduser("~/MOSAIC/data/processed/train.parquet"),
        gw_maps=train_maps,
        out_path="data/processed/train_gw_paired.parquet"
    )
    generate_pairs(
        optical_path=os.path.expanduser("~/MOSAIC/data/processed/val.parquet"),
        gw_maps=val_maps,
        out_path="data/processed/val_gw_paired.parquet"
    )
    generate_pairs(
        optical_path=os.path.expanduser("~/MOSAIC/data/processed/test.parquet"),
        gw_maps=test_maps,
        out_path="data/processed/test_gw_paired.parquet"
    )
