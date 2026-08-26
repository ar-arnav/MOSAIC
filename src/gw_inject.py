import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import astropy_healpix as ah
from gw_processor import GW_data

# 1. Setup paths and get all files
gw_dir = os.path.expanduser("~/MOSAIC/data/raw/GW/")
all_files = glob.glob(os.path.join(gw_dir, "*.fits"))

# 2. Split files ONCE globally
np.random.seed(42)
np.random.shuffle(all_files)
train_maps = all_files[:int(0.8 * len(all_files))]
val_maps = all_files[int(0.8 * len(all_files)):int(0.9 * len(all_files))]
test_maps = all_files[int(0.9 * len(all_files)):]

# 3. The function
def generate_pairs(optical_path, gw_maps, out_path):
    optical_df = pd.read_parquet(optical_path)
    unique_ids = optical_df['group_id'].unique()
    
    results = []
    for map_path in gw_maps:
        batch = np.random.choice(unique_ids, size=50, replace=False)
        gw = GW_data(map_path)
        print(f'Processed {map_path}...')

        for i in batch:
            row = optical_df[optical_df['group_id'] == i].iloc[0]
            label = row['label_class']
            dist = row['dist_mpc']
            
            if label == 1:
                if np.random.rand() < 0.5:
                    pix_idx = np.random.choice(len(gw.pixel_prob), p=gw.pixel_prob)
                    uniq_val = gw.uniq[pix_idx]
                    lvl, px = ah.uniq_to_level_ipix(uniq_val)
                    ns = ah.level_to_nside(lvl)
                    lon, lat = ah.healpix_to_lonlat(px, ns, order='nested')
                    ra = np.degrees(lon).value + np.random.uniform(-0.1, 0.1)
                    dec = np.degrees(lat).value + np.random.uniform(-0.1, 0.1)
                else:
                    ra = np.random.uniform(0, 360)
                    dec = np.random.uniform(-90, 90)
            else:
                if np.random.rand() < 0.5:
                    pix_idx = np.random.choice(len(gw.pixel_prob), p=gw.pixel_prob)
                    uniq_val = gw.uniq[pix_idx]
                    lvl, px = ah.uniq_to_level_ipix(uniq_val)
                    ns = ah.level_to_nside(lvl)
                    lon, lat = ah.healpix_to_lonlat(px, ns, order='nested')
                    ra = np.degrees(lon).value + np.random.uniform(-0.1, 0.1)
                    dec = np.degrees(lat).value + np.random.uniform(-0.1, 0.1)
                else:
                    ra = np.random.uniform(0, 360)
                    dec = np.random.uniform(-90, 90)

            cred_level, dp_dv = gw.evaluate_candidate(ra, dec, dist)
            results.append({'group_id': i, 'label': label, 'gw_cred_level': cred_level, 'gw_dp_dv': dp_dv})

    # Save inside the function
    df_results = pd.DataFrame(results)
    df_results.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

# 4. Execute for all splits
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