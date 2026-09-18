#!/usr/bin/env python3
"""
POSSIS V2 Incremental Injection & Merge Pipeline (StratifiedGroupKFold)
======================================================================
- Scans ONLY new V2 POSSIS files from possis_ztf_clean_v2.
- Uses StratifiedGroupKFold for strict group-aware splitting.
- Splits new POSSIS: 30% integrated into existing train/val/test parquets / 70% isolated holdout.
- Applies correct physical absolute magnitude baseline (MAG_OFFSET = 25.0 for 1 Mpc).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
POSSIS_NEW_DIR = Path("~/MOSAIC/data/possis_ztf_clean_v2").expanduser()
IMPOSTER_ROOT   = Path("~/MOSAIC/data/raw").expanduser()
OUTPUT_DIR      = Path("data/processed").expanduser()
CLASS_ALIAS = {"RRL": "RR Lyrae"}
SCHEMA_COLUMNS = [
    "objectId", "group_id", "jd", "time_day", "fid",
    "magpsf", "sigmapsf", "flux", "flux_err", "diffmaglim",
    "label_class", "subclass", "dist_mpc", "angle_idx",
]

MAX_OBS_DAYS         = 21.0
ZTF_ZP               = 23.9
ZTF_LIM_MAG          = 21.5
MIN_DIST_MPC         = 5.0
MAX_DIST_MPC         = 130.0
MIN_POINTS_KN        = 3
MIN_POINTS_IMP       = 3
MAX_CADENCE_RETRIES  = 40
MAG_OFFSET           = 25.0  # Physically correct baseline for 1 Mpc V2 data
DETECTION_MARGIN_MAG = 0.5
HORIZON_FRAC         = 0.85
T_EXP_JITTER_LO      = -2.0
T_EXP_JITTER_HI      = 5.0


def _seed_worker(pf: str) -> None:
    h = int(hashlib.md5(f"{SEED}:{pf}".encode()).hexdigest()[:8], 16)
    np.random.seed(h)
    random.seed(h)


def sample_volume_weighted_distance(d_min: float, d_max: float) -> float:
    u = np.random.uniform(0.0, 1.0)
    return float((u * (d_max**3 - d_min**3) + d_min**3) ** (1.0 / 3.0))


def apply_flux_noise(mag_true, diffmaglim, zp=ZTF_ZP):
    mag_true = np.asarray(mag_true, dtype=np.float64)
    diffmaglim = np.asarray(diffmaglim, dtype=np.float64)
    flux_true = 10.0 ** (-0.4 * (mag_true - zp))
    flux_5sig = 10.0 ** (-0.4 * (diffmaglim - zp))
    flux_err = flux_5sig / 5.0
    flux_noisy = flux_true + np.random.normal(0.0, flux_err)
    detected = flux_noisy >= 5.0 * flux_err
    mag_noisy = np.full_like(flux_noisy, np.nan, dtype=np.float32)
    sigmapsf = np.full_like(flux_noisy, np.nan, dtype=np.float32)
    if np.any(detected):
        mag_noisy[detected] = zp - 2.5 * np.log10(np.maximum(flux_noisy[detected], 1e-10))
        snr = flux_noisy[detected] / np.maximum(flux_err[detected], 1e-10)
        sigmapsf[detected] = np.sqrt((1.0857 / snr) ** 2 + 0.02 ** 2)
    return mag_noisy, sigmapsf, flux_noisy.astype(np.float32), flux_err.astype(np.float32), detected


def possis_absolute_mag(grp):
    if "magpsf" in grp.columns and grp["magpsf"].notna().any():
        return grp["magpsf"].values.astype(np.float64) - MAG_OFFSET
    flux_val = np.maximum(grp["flux"].values.astype(np.float64), 1e-40)
    m_rel = ZTF_ZP - 2.5 * np.log10(flux_val)
    return m_rel - MAG_OFFSET


def get_imposter_pool() -> List[str]:
    files = []
    if IMPOSTER_ROOT.exists():
        for folder in sorted(IMPOSTER_ROOT.iterdir()):
            if folder.is_dir():
                for f in folder.rglob("*.parquet"):
                    files.append(str(f.resolve()))
    return files


def load_skeleton(fpath):
    df = pd.read_parquet(fpath)
    df = df.drop_duplicates(subset=["time_day", "fid"], keep="first")
    return df.loc[df["fid"] != 3, ["time_day", "fid", "diffmaglim"]].copy()


def process_single_possis_file(pf, skeleton_pool, split_name):
    _seed_worker(pf)
    alerts = []
    n_attempted = n_passed = n_skipped = failed_snr = 0
    distances = []
    possis = pd.read_parquet(pf)
    if "angle_idx" not in possis.columns:
        possis["angle_idx"] = 0

    for angle, grp in possis.groupby("angle_idx"):
        n_attempted += 1
        grp = grp.copy().reset_index(drop=True)
        abs_mag = possis_absolute_mag(grp)
        finite = np.isfinite(abs_mag)
        if finite.sum() < 2:
            n_skipped += 1
            continue
        grp = grp.loc[finite].copy().reset_index(drop=True)
        abs_mag = abs_mag[finite]
        grp["magpsf"] = abs_mag

        M_peak = float(np.min(abs_mag))
        M_eff = M_peak + DETECTION_MARGIN_MAG
        d_horizon = 10.0 ** ((ZTF_LIM_MAG - M_eff - 25.0) / 5.0)
        d_max = min(d_horizon * HORIZON_FRAC, MAX_DIST_MPC)
        if d_max < MIN_DIST_MPC:
            n_skipped += 1
            continue

        dist = sample_volume_weighted_distance(MIN_DIST_MPC, d_max)
        dist_mod = 5.0 * np.log10(dist) + 25.0
        generated = False

        for _ in range(MAX_CADENCE_RETRIES):
            skel = load_skeleton(random.choice(skeleton_pool))
            if len(skel) < MIN_POINTS_KN:
                continue
            t_start = float(skel["time_day"].min())
            t_exp = t_start + random.uniform(T_EXP_JITTER_LO, T_EXP_JITTER_HI)
            obs_window = skel[
                (skel["time_day"] >= t_exp) & (skel["time_day"] <= t_exp + MAX_OBS_DAYS)
            ]
            if len(obs_window) < MIN_POINTS_KN:
                continue

            times, fids, dlims, mags = [], [], [], []
            for fid in obs_window["fid"].unique():
                obs = obs_window[obs_window["fid"] == fid]
                kn = grp[grp["fid"] == fid].sort_values("time_day")
                if len(kn) < 2:
                    continue
                kn_t0 = kn["time_day"].values - kn["time_day"].values.min()
                kn_t = kn_t0 + t_exp
                kn_m = kn["magpsf"].values + dist_mod
                kn_flux = 10.0 ** (-0.4 * (kn_m - ZTF_ZP))
                interp_flux = np.interp(obs["time_day"].values, kn_t, kn_flux, left=0.0, right=0.0)
                valid = interp_flux > 1e-12
                if not np.any(valid):
                    continue
                interp_mag = ZTF_ZP - 2.5 * np.log10(interp_flux[valid])
                valid_mag = (interp_mag >= 10.0) & (interp_mag <= 30.0)
                if not np.any(valid_mag):
                    continue
                times.append(obs["time_day"].values[valid][valid_mag])
                fids.append(obs["fid"].values[valid][valid_mag])
                dlims.append(obs["diffmaglim"].values[valid][valid_mag])
                mags.append(interp_mag[valid_mag])

            if not mags:
                continue
            t_arr = np.concatenate(times)
            fid_arr = np.concatenate(fids)
            dlim_arr = np.concatenate(dlims)
            mag_arr = np.concatenate(mags)
            mag_obs, sig_obs, flux_obs, ferr_obs, det = apply_flux_noise(mag_arr, dlim_arr)
            if det.sum() < MIN_POINTS_KN:
                failed_snr += 1
                continue

            obj_id = f"KN_{Path(pf).stem}_a{int(angle)}"
            alerts.append(pd.DataFrame({
                "objectId": obj_id, "group_id": obj_id,
                "jd": t_arr[det] + 2458000.0, "time_day": t_arr[det],
                "fid": fid_arr[det], "magpsf": mag_obs[det], "sigmapsf": sig_obs[det],
                "flux": flux_obs[det], "flux_err": ferr_obs[det], "diffmaglim": dlim_arr[det],
                "label_class": 1, "subclass": "Kilonova",
                "dist_mpc": np.float32(dist), "angle_idx": np.int8(angle),
            })[SCHEMA_COLUMNS])
            n_passed += 1
            distances.append(dist)
            generated = True
            break

        if not generated:
            n_skipped += 1

    return alerts, n_attempted, n_passed, n_skipped, distances, failed_snr


def inject_possis(possis_files, skeleton_pool, split_name, n_workers=4):
    if not possis_files:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    all_alerts = []
    total_attempted = total_passed = total_skipped = total_failed_snr = 0
    all_distances = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_single_possis_file, pf, skeleton_pool, split_name)
            for pf in possis_files
        ]
        for fut in as_completed(futures):
            alerts, n_att, n_pass, n_skip, dists, fail_snr = fut.result()
            all_alerts.extend(alerts)
            total_attempted += n_att
            total_passed += n_pass
            total_skipped += n_skip
            all_distances.extend(dists)
            total_failed_snr += fail_snr
    rate = 100.0 * total_passed / max(total_attempted, 1)
    print(f"    {split_name.upper()} KNs: {total_passed}/{total_attempted} generated "
          f"({rate:.1f}%) | {total_skipped} skipped (SNR failures: {total_failed_snr})")
    if all_distances:
        print(f"      mean dist = {np.mean(all_distances):.1f} Mpc  "
              f"median = {np.median(all_distances):.1f} Mpc")
    if not all_alerts:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.concat(all_alerts, ignore_index=True)


def main(n_workers=4):
    print("=" * 60)
    print("  POSSIS V2 SGKF Incremental Merge & Holdout Pipeline")
    print("=" * 60)

    if not POSSIS_NEW_DIR.exists():
        raise FileNotFoundError(f"New POSSIS directory not found: {POSSIS_NEW_DIR}")

    new_possis_files = sorted(list(POSSIS_NEW_DIR.rglob("*.parquet")))
    print(f"  Found {len(new_possis_files)} new POSSIS V2 files.")

    # Build metadata registry for grouping
    df_new_meta = pd.DataFrame([
        {"filepath": str(f.resolve()), "group": f.stem.split('_a')[0], "subclass": "Kilonova"} 
        for f in new_possis_files
    ])

    unique_groups = df_new_meta["group"].unique()
    
    # --- 1. SPLIT NEW POSSIS 30% / 70% BY GROUP ---
    np.random.seed(SEED)
    np.random.shuffle(unique_groups)
    split_idx = int(len(unique_groups) * 0.30)
    groups_30 = unique_groups[:split_idx]
    groups_70 = unique_groups[split_idx:]

    df_30_meta = df_new_meta[df_new_meta["group"].isin(groups_30)].copy()
    df_70_meta = df_new_meta[df_new_meta["group"].isin(groups_70)].copy()

    files_70 = df_70_meta["filepath"].tolist()

    print(f"  Split Allocation:")
    print(f"    -> 30% Integration Pool: {len(df_30_meta)} files across {len(groups_30)} models")
    print(f"    -> 70% Holdout Pool:     {len(df_70_meta)} files across {len(groups_70)} models")

    imposter_pool = get_imposter_pool()

    # --- 2. PARTITION 30% POOL VIA STRATIFIED GROUP K-FOLD ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(
        sgkf.split(X=df_30_meta, y=df_30_meta["subclass"], groups=df_30_meta["group"])
    )
    df_train_val = df_30_meta.iloc[train_val_idx]
    df_test = df_30_meta.iloc[test_idx]

    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf_val.split(X=df_train_val, y=df_train_val["subclass"], groups=df_train_val["group"])
    )
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    f_train = df_train["filepath"].tolist()
    f_val = df_val["filepath"].tolist()
    f_test = df_test["filepath"].tolist()

    print(f"    -> Train Integration Files: {len(f_train)}")
    print(f"    -> Val Integration Files:   {len(f_val)}")
    print(f"    -> Test Integration Files:  {len(f_test)}")

    # Strict Leakage Verification
    assert set(df_train["group"]).isdisjoint(set(df_val["group"]))
    assert set(df_train["group"]).isdisjoint(set(df_test["group"]))
    assert set(df_val["group"]).isdisjoint(set(df_test["group"]))
    print("    -> SGKF group leakage assertions passed successfully.")

    # --- 3. INJECT & MERGE INTO EXISTING DATASETS ---
    print("\n  --- Injecting 30% subset for pipeline integration ---")
    new_train_kn = inject_possis(f_train, imposter_pool, "train_integration", n_workers=n_workers)
    new_val_kn   = inject_possis(f_val, imposter_pool, "val_integration", n_workers=n_workers)
    new_test_kn  = inject_possis(f_test, imposter_pool, "test_integration", n_workers=n_workers)

    splits_data = {
        "train": new_train_kn,
        "val": new_val_kn,
        "test": new_test_kn,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, new_kn_df in splits_data.items():
        parquet_path = OUTPUT_DIR / f"{split_name}.parquet"
        print(f"\n  Merging into existing {split_name}.parquet...")
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            print(f"    Loaded existing rows: {len(existing_df):,}")
            combined_df = pd.concat([existing_df, new_kn_df], ignore_index=True)
        else:
            print(f"    No existing file found for {split_name}. Creating new.")
            combined_df = new_kn_df

        combined_df.to_parquet(parquet_path, index=False)
        print(f"    -> Saved updated {split_name}.parquet with {len(combined_df):,} total rows.")

    # --- 4. PROCESS 70% INTO ISOLATED HOLDOUT FOLDER ---
    print("\n  --- Generating 70% Isolated E2E Holdout Set ---")
    holdout_df = inject_possis(files_70, imposter_pool, "holdout", n_workers=n_workers)
    
    holdout_dir = OUTPUT_DIR / "v2_e2e_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_out_path = holdout_dir / "kilonova_v2_holdout_70.parquet"
    holdout_df.to_parquet(holdout_out_path, index=False)
    
    print(f"  Holdout Summary: {holdout_df['group_id'].nunique()} unique models, {len(holdout_df):,} rows")
    print(f"  -> Saved holdout to: {holdout_out_path}")
    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    main(n_workers=args.workers)