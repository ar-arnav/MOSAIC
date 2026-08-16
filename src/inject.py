import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold 

def scale_distance(df, reference_distance: int = 40):
    max_distance = 200
    min_distance = 10

    # Uniformly sample in 3D volume, then convert back to distance
    target_distance = (np.random.uniform(0, 1) * ((max_distance)**3 - (min_distance)**3) + (min_distance)**3)**(1/3)
    delta_m = 5 * np.log10(target_distance / reference_distance)

    df = df.copy()
    df['magpsf'] += delta_m
    df['flux'] *= 10**(-0.4 * delta_m)
    df['flux_err'] *= 10**(-0.4 * delta_m)

    return df

def generate_synthetic_kilonovae(kn_path: str, ztf_skeleton_path: str) -> pd.DataFrame:
    kn_path = Path(os.path.expanduser(kn_path))
    ztf_skeleton_path = Path(os.path.expanduser(ztf_skeleton_path))
    
    synthetic_kn = []
    synthetic_files = list(ztf_skeleton_path.rglob("*.parquet"))

    if not synthetic_files:
        raise ValueError(f"No ZTF skeleton files found in {ztf_skeleton_path}")

    for file in kn_path.rglob("*.parquet"):
        df = pd.read_parquet(file)

        for angle, angle_group in df.groupby('angle_idx'):
            scaled_df = scale_distance(angle_group)
            max_kn_time = scaled_df['time_day'].max()

            chosen = np.random.choice(synthetic_files)
            synthetic_df = pd.read_parquet(chosen)

            trimmed_ztf_df = synthetic_df[synthetic_df['time_day'] <= max_kn_time].copy()

            if len(trimmed_ztf_df) == 0:
                continue

            synthetic_alert = trimmed_ztf_df.copy()
            interpolated_mags = np.empty_like(trimmed_ztf_df['magpsf'].values)

            for fid in trimmed_ztf_df['fid'].unique():
                synthetic_mask = trimmed_ztf_df['fid'] == fid
                possis_mask = scaled_df['fid'] == fid

                if np.sum(synthetic_mask) > 1 and np.sum(possis_mask) > 1:
                    # CRITICAL FIX: Sort POSSIS data by time to satisfy np.interp monotonicity requirement
                    sorted_possis = scaled_df.loc[possis_mask].sort_values('time_day')
                    
                    interpolated_mags[synthetic_mask] = np.interp(
                        trimmed_ztf_df.loc[synthetic_mask, 'time_day'].values,
                        sorted_possis['time_day'].values,
                        sorted_possis['magpsf'].values,
                        left=np.nan, right=np.nan
                    )
                else:
                    interpolated_mags[synthetic_mask] = np.nan

            # Assign the interpolated magnitudes
            synthetic_alert['magpsf'] = interpolated_mags
            
            # CRITICAL FIX: Drop rows where POSSIS didn't simulate that filter band (e.g., ZTF 'i' band)
            synthetic_alert.dropna(subset=['magpsf'], inplace=True)

            # If too much of the light curve was dropped, skip this object
            if len(synthetic_alert) < 3: # 3 is your min_points
                continue

            # Explicitly match the exact schema columns
            synthetic_alert['angle_idx'] = np.int8(-1) 
            synthetic_alert['label_class'] = 1
            synthetic_alert['subclass'] = 'Kilonovae'
            synthetic_alert['group_id'] = f"KN_{file.stem}_angle_{angle}"
            
            ZP = 23.9  
            old_flux = synthetic_alert['flux'].copy()
            synthetic_alert['flux'] = 10**(-0.4 * (synthetic_alert['magpsf'] - ZP))
            
            flux_ratio = np.where(old_flux > 0, synthetic_alert['flux'] / old_flux, 1.0)
            synthetic_alert['flux_err'] *= flux_ratio

            synthetic_kn.append(synthetic_alert)

    return pd.concat(synthetic_kn, ignore_index=True)

imposter_config = {
    'AGN': '~/MOSAIC/data/raw/AGN',
    'Blazar': '~/MOSAIC/data/raw/Blazar',
    'CV/Nova': '~/MOSAIC/data/raw/CV/Nova',
    'E': '~/MOSAIC/data/raw/E',
    'LPV': '~/MOSAIC/data/raw/LPV',
    'QSO': '~/MOSAIC/data/raw/QSO',
    'RRL': '~/MOSAIC/data/raw/RRL',
    'SLSN': '~/MOSAIC/data/raw/SLSN',
    'SNIa': '~/MOSAIC/data/raw/SNIa',
    'SNII': '~/MOSAIC/data/raw/SNII',
    'SNIbc': '~/MOSAIC/data/raw/SNIbc',
    'YSO': '~/MOSAIC/data/raw/YSO',
}

def load_imposter_dataset(imposter_config: dict, max_days: float = 30.0, min_points: int = 2) -> pd.DataFrame:
    imposter_alerts = []

    for subclass, path_str in imposter_config.items():
        folder = Path(os.path.expanduser(path_str))
        if not folder.exists():
            continue

        for file in folder.rglob("*.parquet"):
            df = pd.read_parquet(file)

            trimmed_df = df[df['time_day'] <= max_days].copy()

            if len(trimmed_df) < min_points:
                continue

            trimmed_df['label_class'] = 0
            trimmed_df['subclass'] = subclass
            trimmed_df['group_id'] = f"{subclass}_{file.stem}"

            imposter_alerts.append(trimmed_df)

    if not imposter_alerts:
        raise ValueError("No valid imposter files were found in the specified paths.")

    return pd.concat(imposter_alerts, ignore_index=True)

if __name__ == "__main__":
    print("Generating synthetic Kilonovae...")
    final_synthetic_dataset = generate_synthetic_kilonovae(
        kn_path="~/MOSAIC/data/possis_ztf_clean",
        ztf_skeleton_path="~/MOSAIC/data/raw/SNIbc"
    )

    print("Loading Imposter Dataset...")
    final_imposter_dataset = load_imposter_dataset(imposter_config)

    print("Combining datasets...")
    master_dataset = pd.concat([final_synthetic_dataset, final_imposter_dataset], ignore_index=True)

    # Remove exact duplicate light curve points
    master_dataset = master_dataset.drop_duplicates(
        subset=['group_id', 'time_day', 'fid'], 
        keep='first'
    ).reset_index(drop=True)

    # Stratified Group K-Fold Splitting
    unique_objects = master_dataset[['group_id', 'label_class']].drop_duplicates().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,  random_state=42)

    train_val_idx, test_idx = next(splitter.split(
        X=np.zeros(len(unique_objects)), 
        y=unique_objects['label_class'], 
        groups=unique_objects['group_id']
    ))

    train_val_df = master_dataset[master_dataset['group_id'].isin(unique_objects.iloc[train_val_idx]['group_id'])].reset_index(drop=True)
    test_df = master_dataset[master_dataset['group_id'].isin(unique_objects.iloc[test_idx]['group_id'])].reset_index(drop=True)

    # Second Split: Train (80%) vs Val (20%)
    tv_unique = train_val_df[['group_id', 'label_class']].drop_duplicates().reset_index(drop=True)
    
    train_idx, val_idx = next(splitter.split(
        X=np.zeros(len(tv_unique)), 
        y=tv_unique['label_class'], 
        groups=tv_unique['group_id']
    ))

    train_df = train_val_df[train_val_df['group_id'].isin(tv_unique.iloc[train_idx]['group_id'])].reset_index(drop=True)
    val_df = train_val_df[train_val_df['group_id'].isin(tv_unique.iloc[val_idx]['group_id'])].reset_index(drop=True)

    # Final Export
    print("Saving parquet files...")
    train_df.to_parquet('data/processed/train.parquet', index=False)
    val_df.to_parquet('data/processed/val.parquet', index=False)
    test_df.to_parquet('data/processed/test.parquet', index=False)

    print(f"Done! Train: {train_df['group_id'].nunique()} objects | Val: {val_df['group_id'].nunique()} objects | Test: {test_df['group_id'].nunique()} objects")