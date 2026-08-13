import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Getting KN data to split from raw parquet files

kn_folder = Path("data/possis_ztf_clean")

kn_files = list(kn_folder.glob("*.parquet"))

df_kn_list = []

for file in kn_files:
    df = pd.read_parquet(file)
    df_kn_list.append(df)

df_kn = pd.concat(df_kn_list, ignore_index=True)

# Grouping objects based on their Object_ID (Viewing angle) and splitting them into train and test sets using StratifiedGroupKFold

df_kn["group"] = df_kn["objectId"]
df_kn["label"] = 'Kilonovae'

print("Total number of Kilonovae objects:", df_kn["group"].nunique())

# Getting Imposter data to split from raw parquet files
imposter_root = Path("data/raw")

imposter_config = {
    "AGN": {"folder": "AGN", "limit": 25000},
    "E": {"folder": "E", "limit": 20000},
    "CV": {"folder": "CV", "limit": 15000},
    "LPV": {"folder": "LPV", "limit": 10000},
    "RR Lyrae": {"folder": "RR_Lyrae", "limit": 10000},
    "SNIa": {"folder": "SNIa", "limit": 8000},
    "SNIbc": {"folder": "SNIbc", "limit": 8000},
    "SNII": {"folder": "SNIIn", "limit": 5000},
    "YSO": {"folder": "YSO", "limit": 4000}
}

df_imposter_dict = {}

for class_name, conf in imposter_config.items():
    class_folder = imposter_root / conf["folder"]

    files = sorted(list(class_folder.glob("*.parquet")))

# Assigning group and label columns to each imposter dataframe and storing them in a dictionary

    dfs = []
    for file in files:
        df = pd.read_parquet(file)

        df["group"] = f"{class_name}_{file.stem}"
        df["label"] = class_name
        df["source_file"] = str(file)

        dfs.append(df)

    if dfs:
        df_imposter = pd.concat(dfs, ignore_index=True)
        df_imposter_dict[class_name] = df_imposter
        print(f"Total number of {class_name} objects:", df_imposter["group"].nunique())

# Degrading the POSSIS KN data to make it more realistic and similar to the imposters

# Distance Scaling
def scale_distance(df:pd.DataFrame, target_distance_mpc: float, reference_distance_mpc: float = 40) -> pd.DataFrame:
    delta_m = 5 * np.log10(target_distance_mpc / reference_distance_mpc)
    df_scaled = df.copy()

    df_scaled["magpsf"] += delta_m

    flux_scale_factor = 10 ** (-0.4 * delta_m)
    df_scaled["flux"] *= flux_scale_factor

    df_scaled["distance_mpc"] = target_distance_mpc
    return df_scaled

# Injecting Noise into the POSSIS KN data to simulate realistic observational conditions
def inject_noise(df: pd.DataFrame, snr_threshold: float = 3.0, seed: int | None = None, rng : np.random.Generator | None = None) -> pd.DataFrame:

    """ If no random number generator is provided, create one with the given seed.
        Thus provide either rng or seed not both. If both are None, a new random generator will be created with a random seed. """

    if rng is None:
        rng = np.random.default_rng(seed)

    magpsf = df["magpsf"].to_numpy(dtype=np.float32, copy=True)
    diffmaglim = df["diffmaglim"].to_numpy(dtype=np.float32, copy=False)
    flux = df["flux"].to_numpy(dtype=np.float32, copy=True)

    snr = 5 * np.power(10, -0.4 * (magpsf - diffmaglim))

    sigmapsf = 1.0857/np.maximum(snr, 1e-5, out=snr)
    sigmapsf = sigmapsf.astype(np.float32, copy=False)

    flux_err = (np.log(10.0)/2.5) * sigmapsf * flux

    gaussian_noise = rng.normal(loc=0.0, scale = flux_err).astype(np.float32, copy=False)
    flux += gaussian_noise   

    snr_obs = np.zeros_like(snr, dtype=np.float32)
    valid_err = flux_err > 0
    np.divide(flux, flux_err, out=snr_obs, where=valid_err)

    pos_mask = flux > 0
    valid_snr_pos = pos_mask & (snr_obs > 0)

    magpsf[valid_snr_pos] = diffmaglim[valid_snr_pos] -2.5 * np.log10(snr_obs[valid_snr_pos]/5.0).astype(np.float32)

    detected_mask = snr_obs >= snr_threshold

    df_detected = df[detected_mask].copy()

    df_detected["magpsf"] = magpsf[detected_mask]
    df_detected["sigmapsf"] = sigmapsf[detected_mask]
    df_detected["flux"] = flux[detected_mask]
    df_detected["flux_err"] = flux_err[detected_mask]

    return df_detected.reset_index(drop=True)


# Running the functions to get scaled and noisy KN data


# Adding scaled distnace and recalculating flux_scale and magnitude factor for the KN data to simulate realistic observational conditions.

# First adding a Uniform-in-Volume distribution to model the proper target distance distribution for Kilonovae. The target distance is set between 10-200 Mpc.

d_min = 10
d_max = 200

d_vol = (np.random.uniform(0, 1) *(d_max**3 - d_min**3) + d_min**3)**(1/3)

df_kn_scaled = scale_distance(df_kn, target_distance_mpc=d_vol, reference_distance_mpc=40)


# Adding noise to the scaled KN data with a SNR threshold of 3.0 and a random seed of 42 for reproducibility
df_kn_noisy = inject_noise(df_kn_scaled, snr_threshold=3.0, seed=42)

# Combining data to be split into train, validate and test sets. The KN data is combined with the imposter data to create a single dataframe for splitting.
df_all = pd.concat([df_kn_noisy] + list(df_imposter_dict.values()), ignore_index=True)

object_meta = df_all[["group", "label"]].drop_duplicates().reset_index(drop=True)

# StratifiedGroupKFold implementation

sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
 
train_val_idx, test_idx = next(sgkf.split(
    X=object_meta["group"],
    y=object_meta["label"], 
    groups=object_meta["group"]
    ))

train_val_objects = set(object_meta.iloc[train_val_idx]["group"])
test_objects = set(object_meta.iloc[test_idx]["group"])

df_tv_meta = object_meta[
    object_meta["group"].isin(train_val_objects)
].reset_index(drop=True)

sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)

train_idx_sub, val_idx_sub = next(
    sgkf_val.split(
        X=df_tv_meta["group"],
        y=df_tv_meta["label"],
        groups=df_tv_meta["group"],
    )
)

train_objects = set(df_tv_meta.iloc[train_idx_sub]["group"])
val_objects = set(df_tv_meta.iloc[val_idx_sub]["group"])

# Final split output of all data into train, validate and test sets based on the object groups obtained from the StratifiedGroupKFold splits.
df_train = df_all[df_all["group"].isin(train_objects)].reset_index(drop=True)

df_val = df_all[df_all["group"].isin(val_objects)].reset_index(drop=True)

df_test = df_all[df_all["group"].isin(test_objects)].reset_index(drop=True)

