import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import astropy_healpix as ah

from gw_processor import GW_data

train_path = os.path.expanduser("~/MOSAIC/data/processed/train.parquet")
# val_path = os.path.expanduser("~/MOSAIC/data/processed/val.parquet")
# test_path = os.path.expanduser("~/MOSAIC/data/processed/test.parquet")
path = os.path.expanduser("~/MOSAIC/data/raw/GW/")
files = glob.glob(os.path.join(path, "*.fits"))


np.random.seed(42)

np.random.shuffle(files)
train_map = files[:int(0.8 * len(files))]
val_map = files[int(0.8 * len(files)):int(0.9 * len(files))]
test_map = files[int(0.9 * len(files)):]

train = pd.read_parquet(train_path)

unique = train['group_id'].unique()

results = []
for map in train_map:
    batch = np.random.choice(unique, size=50, replace=False)
    gw = GW_data(map)
    print(f'Processed {map}...')

    for i in batch:
        row = train[train['group_id'] == i].iloc[0]
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


df_results = pd.DataFrame(results)
df_results.to_parquet('data/processed/train_gw_paired.parquet', index=False)


# val = pd.read_parquet(val_path)
# test = pd.read_parquet(test_path)
