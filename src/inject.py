#!/usr/bin/env python3
"""
POSSIS -> ZTF Kilonova Injection Pipeline
=========================================
- magpsf-first abs mag (MAG_OFFSET on magpsf only)
- Yield knobs: more retries, softer margin, horizon frac
- ToO jitter for KNe
- Duration matching: imposters truncated to short spans like KN arcs
  (fixes AUC≈1 from duration-only leak)
- Detections-only both classes, StratifiedGroupKFold
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
POSSIS_DIR    = Path("~/MOSAIC/data/possis_ztf_clean").expanduser()
IMPOSTER_ROOT = Path("~/MOSAIC/data/raw").expanduser()
OUTPUT_DIR    = Path("data/processed").expanduser()
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
MAG_OFFSET           = 55.33
DETECTION_MARGIN_MAG = 0.5
HORIZON_FRAC         = 0.85
T_EXP_JITTER_LO      = -2.0
T_EXP_JITTER_HI      = 5.0

# Imposter duration matching (days of detections after first)
# Match post-hoc fix: KN arcs are ~0.5 d mean; keep imposters in the same range.
IMP_DUR_LO = 0.25
IMP_DUR_HI = 2.5


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


def compute_imposter_photometry(df, zp=ZTF_ZP):
    if (
        "flux" in df.columns and "flux_err" in df.columns and "sigmapsf" in df.columns
        and df["flux"].notna().all() and df["flux_err"].notna().all() and df["sigmapsf"].notna().all()
    ):
        return (
            df["sigmapsf"].values.astype(np.float32),
            df["flux"].values.astype(np.float32),
            df["flux_err"].values.astype(np.float32),
        )
    mag = np.clip(np.asarray(df["magpsf"].values, dtype=np.float64), 10.0, 30.0)
    diffmaglim = np.clip(np.asarray(df["diffmaglim"].values, dtype=np.float64), 10.0, 25.0)
    flux = 10.0 ** (-0.4 * (mag - zp))
    flux_5sig = 10.0 ** (-0.4 * (diffmaglim - zp))
    flux_err = flux_5sig / 5.0
    snr = np.maximum(flux / np.maximum(flux_err, 1e-10), 1e-3)
    sigmapsf = np.sqrt((1.0857 / snr) ** 2 + 0.02 ** 2)
    return sigmapsf.astype(np.float32), flux.astype(np.float32), flux_err.astype(np.float32)


def possis_absolute_mag(grp):
    if "magpsf" in grp.columns and grp["magpsf"].notna().any():
        return grp["magpsf"].values.astype(np.float64) - MAG_OFFSET
    flux_val = np.maximum(grp["flux"].values.astype(np.float64), 1e-40)
    m_rel = ZTF_ZP - 2.5 * np.log10(flux_val)
    return m_rel - MAG_OFFSET


def build_registry():
    records = []
    if IMPOSTER_ROOT.exists():
        for folder in sorted(IMPOSTER_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            sub = CLASS_ALIAS.get(folder.name, folder.name)
            for f in folder.rglob("*.parquet"):
                records.append({
                    "filepath": str(f.resolve()),
                    "subclass": sub,
                    "group": f"{sub}__{f.stem}",
                    "type": "imposter",
                })
    if POSSIS_DIR.exists():
        for f in POSSIS_DIR.rglob("*.parquet"):
            records.append({
                "filepath": str(f.resolve()),
                "subclass": "Kilonova",
                "group": f"Kilonova__{f.stem}",
                "type": "possis",
            })
    if not records:
        raise FileNotFoundError("No input Parquet files found.")
    return pd.DataFrame(records)


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


def load_imposters(file_list, subclass_map):
    """
    Same front-loaded outer window as before, then:
      1) 5σ detections only
      2) truncate detections to a short random span [IMP_DUR_LO, IMP_DUR_HI]
         so duration distribution matches KN arcs (fixes duration leak).
    """
    alerts = []
    skipped_bad_format = skipped_low_snr = skipped_short = 0

    for fpath in file_list:
        p = Path(fpath)
        if not p.exists():
            skipped_bad_format += 1
            continue

        df = pd.read_parquet(p)
        df = df.drop_duplicates(subset=["time_day", "fid"], keep="first")
        required = {"time_day", "fid", "magpsf", "diffmaglim"}
        if not required.issubset(df.columns):
            skipped_bad_format += 1
            continue

        # Outer 21-day pool (same as KN max window)
        t_start = float(df["time_day"].min())
        df = df[
            (df["time_day"] >= t_start)
            & (df["time_day"] <= t_start + MAX_OBS_DAYS)
            & (df["fid"] != 3)
        ].copy()
        if len(df) < MIN_POINTS_IMP:
            skipped_low_snr += 1
            continue

        sig, flux, ferr = compute_imposter_photometry(df)
        snr = np.where(ferr > 0, flux / np.maximum(ferr, 1e-10), 0.0)
        detections = snr >= 5.0
        if detections.sum() < MIN_POINTS_IMP:
            skipped_low_snr += 1
            continue

        df = df.loc[detections].copy()
        df["sigmapsf"] = sig[detections]
        df["flux"] = flux[detections]
        df["flux_err"] = ferr[detections]

        # --- duration match: short random span from first detection ---
        t0 = float(df["time_day"].min())
        span = random.uniform(IMP_DUR_LO, IMP_DUR_HI)
        df = df[df["time_day"] <= t0 + span].copy()
        if len(df) < MIN_POINTS_IMP:
            skipped_short += 1
            continue

        df["label_class"] = 0
        df["subclass"] = subclass_map[fpath]
        df["dist_mpc"] = np.float32(np.nan)
        df["angle_idx"] = np.int8(-1)
        df["group_id"] = f"{df['subclass'].iloc[0]}_{p.stem}"
        df["objectId"] = df["group_id"]
        if "jd" not in df.columns:
            df["jd"] = df["time_day"] + 2458000.0

        alerts.append(df[SCHEMA_COLUMNS])

    print(
        f"    Imposters: {len(alerts)} loaded | "
        f"{skipped_low_snr} low-SNR | {skipped_short} too-short after dur-match | "
        f"{skipped_bad_format} bad"
    )
    if not alerts:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.concat(alerts, ignore_index=True)


def main(n_workers=4):
    print("=" * 60)
    print("  POSSIS Injection (magpsf-first + duration match)")
    print("=" * 60)
    meta = build_registry()
    print(
        f"  Indexed {len(meta)} files "
        f"({(meta.type == 'possis').sum()} POSSIS, {(meta.type == 'imposter').sum()} imposters)"
    )
    print(
        f"  Imposter subclasses: "
        f"{sorted(meta.loc[meta.type == 'imposter', 'subclass'].unique())}"
    )
    print(f"  Imp duration match: [{IMP_DUR_LO}, {IMP_DUR_HI}] days")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(
        sgkf.split(X=meta, y=meta["subclass"], groups=meta["group"])
    )
    train_val, test_meta = meta.iloc[train_val_idx], meta.iloc[test_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf2.split(X=train_val, y=train_val["subclass"], groups=train_val["group"])
    )
    train_meta, val_meta = train_val.iloc[train_idx], train_val.iloc[val_idx]

    for name, m in [("Train", train_meta), ("Val", val_meta), ("Test", test_meta)]:
        print(
            f"  {name:5s}: {len(m):5d} files  "
            f"({(m.type == 'imposter').sum()} imp, {(m.type == 'possis').sum()} POSSIS)"
        )
    assert set(train_meta.group).isdisjoint(val_meta.group)
    assert set(train_meta.group).isdisjoint(test_meta.group)
    assert set(val_meta.group).isdisjoint(test_meta.group)
    print("  zero group leakage OK")

    subclass_map = dict(zip(meta["filepath"], meta["subclass"]))
    datasets = {}

    for split, m in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        print(f"\n  --- {split.upper()} ---")
        possis_files = m[m["type"] == "possis"]["filepath"].tolist()
        imp_files = m[m["type"] == "imposter"]["filepath"].tolist()
        kn_df = inject_possis(possis_files, imp_files, split, n_workers=n_workers)
        imp_df = load_imposters(imp_files, subclass_map)
        combined = pd.concat([kn_df, imp_df], ignore_index=True)
        datasets[split] = combined

        n_kn = combined.loc[combined.label_class == 1, "group_id"].nunique()
        n_imp = combined.loc[combined.label_class == 0, "group_id"].nunique()
        print(f"  Summary: {n_kn} kilonovae, {n_imp} imposters, {len(combined):,} rows")
        assert combined["magpsf"].notna().all()
        assert combined["sigmapsf"].notna().all()

        # Duration / peak sanity
        def _obj_stats(sub):
            rows = []
            for _, g in sub.groupby("group_id"):
                t = g["time_day"].values
                m = g["magpsf"].values
                rows.append({
                    "dur": float(t.max() - t.min()) if len(t) > 1 else 0.0,
                    "peak": float(m.min()),
                    "n": len(g),
                })
            return pd.DataFrame(rows)

        kn_s = _obj_stats(combined[combined.label_class == 1])
        imp_s = _obj_stats(combined[combined.label_class == 0])
        if len(kn_s):
            print(
                f"  KN  peak mean={kn_s.peak.mean():.2f}  dur mean={kn_s.dur.mean():.2f}  "
                f"n mean={kn_s.n.mean():.1f}"
            )
        if len(imp_s):
            print(
                f"  Imp peak mean={imp_s.peak.mean():.2f}  dur mean={imp_s.dur.mean():.2f}  "
                f"n mean={imp_s.n.mean():.1f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in datasets.items():
        out = OUTPUT_DIR / f"{split}.parquet"
        df.to_parquet(out, index=False)
        print(f"  -> {out} ({len(df):,} rows)")
    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    main(n_workers=args.workers)
