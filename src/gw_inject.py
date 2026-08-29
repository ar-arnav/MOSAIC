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

        # Pre-normalize pixel probabilities for sampling
        pixel_prob_normalized = gw.pixel_prob / gw.pixel_prob.sum()
        
        # Create weighted probability for pixels with VALID distance info
        # This ensures KN distance sampling always gets valid mu/sigma
        valid_dist_mask = gw.has_valid_distance
        if valid_dist_mask.sum() > 0:
            valid_pixel_prob = pixel_prob_normalized * valid_dist_mask
            valid_pixel_prob = valid_pixel_prob / valid_pixel_prob.sum()
        else:
            valid_pixel_prob = pixel_prob_normalized

        for i in batch:
            row = optical_df[optical_df['objectId'] == i].iloc[0]
            label = row['label_class']
            
            if label == 1:
                # KILONOVA: True GW Counterpart
                # Sample from pixels with valid distance info
                pix_idx = np.random.choice(len(valid_pixel_prob), p=valid_pixel_prob)
                uniq_val = gw.uniq[pix_idx]
                lvl, px = ah.uniq_to_level_ipix(uniq_val)
                ns = ah.level_to_nside(lvl)
                lon, lat = ah.healpix_to_lonlat(px, ns, order='nested')
                
                ra = np.degrees(lon).value + np.random.uniform(-0.05, 0.05)
                dec = np.degrees(lat).value + np.random.uniform(-0.05, 0.05)
                
                # Distance from GW posterior
                dist = np.random.normal(gw.distmu[pix_idx], gw.distsig[pix_idx])
                dist = max(5.0, dist)
                
            else:
                # IMPOSTER: Inside GW contour (FIXED - no spatial leakage)
                # Sample from full GW sky map (not just valid distance pixels)
                pix_idx = np.random.choice(len(pixel_prob_normalized), p=pixel_prob_normalized)
                uniq_val = gw.uniq[pix_idx]
                lvl, px = ah.uniq_to_level_ipix(uniq_val)
                ns = ah.level_to_nside(lvl)
                lon, lat = ah.healpix_to_lonlat(px, ns, order='nested')
                
                ra = np.degrees(lon).value + np.random.uniform(-0.05, 0.05)
                dec = np.degrees(lat).value + np.random.uniform(-0.05, 0.05)
                
                # Independent distance (NOT from GW posterior)
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