#!/usr/bin/env python3
"""
Quick sanity check on MOSAIC imposters vs Kilonovae.
Run from ~/MOSAIC:
  python src/check_imposters.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("~/MOSAIC").expanduser()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
V2 = ROOT / "data" / "possis_ztf_clean_v2"

SCHEMA_EXPECTED = [
    "objectId", "group_id", "jd", "time_day", "fid",
    "magpsf", "sigmapsf", "flux", "flux_err", "diffmaglim",
    "label_class", "subclass", "dist_mpc", "angle_idx",
]

def load_one(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  FAIL read {path.name}: {e}")
        return None

def summarize_object(df: pd.DataFrame, name: str):
    print(f"\n=== {name} ===")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    missing = [c for c in SCHEMA_EXPECTED if c not in df.columns]
    extra   = [c for c in df.columns if c not in SCHEMA_EXPECTED]
    if missing:
        print(f"  ⚠ MISSING expected cols: {missing}")
    if extra:
        print(f"  extra cols: {extra}")

    # dtypes of key columns
    for c in ["time_day", "fid", "magpsf", "sigmapsf", "flux", "label_class", "dist_mpc", "angle_idx"]:
        if c in df.columns:
            print(f"  {c:12s} dtype={df[c].dtype}  nulls={df[c].isna().sum()}")

    # object-level stats (if group_id / objectId present)
    key = "group_id" if "group_id" in df.columns else "objectId"
    if key in df.columns:
        n_obj = df[key].nunique()
        print(f"  unique objects: {n_obj}")

        # duration & peak per object
        rows = []
        for oid, g in df.groupby(key):
            t = g["time_day"].values
            m = g["magpsf"].values
            rows.append({
                "n": len(g),
                "dur": float(t.max() - t.min()) if len(t) > 1 else 0.0,
                "peak": float(np.nanmin(m)),
                "mean_m": float(np.nanmean(m)),
            })
        st = pd.DataFrame(rows)
        print(f"  n_points  mean={st.n.mean():.1f}  med={st.n.median():.1f}")
        print(f"  duration  mean={st.dur.mean():.2f} d  med={st.dur.median():.2f} d")
        print(f"  peak mag  mean={st.peak.mean():.2f}  med={st.peak.median():.2f}")
        print(f"  mean mag  mean={st.mean_m.mean():.2f}")

    # label / dist checks
    if "label_class" in df.columns:
        print(f"  label_class value counts:\n{df['label_class'].value_counts().to_string()}")
    if "dist_mpc" in df.columns:
        n_nan = df["dist_mpc"].isna().sum()
        n_finite = df["dist_mpc"].notna().sum()
        print(f"  dist_mpc: {n_nan} NaN  |  {n_finite} finite")
        if n_finite > 0:
            print(f"    finite range: {df['dist_mpc'].min():.1f} – {df['dist_mpc'].max():.1f} Mpc")
    if "subclass" in df.columns:
        print(f"  subclasses: {sorted(df['subclass'].dropna().unique().tolist())}")

def main():
    print("=" * 60)
    print("  MOSAIC Imposter + KN schema / stats check")
    print("=" * 60)

    # 1. Existing processed train (if present)
    for split in ["train", "val", "test"]:
        p = PROCESSED / f"{split}.parquet"
        if p.exists():
            df = load_one(p)
            if df is not None:
                kn  = df[df["label_class"] == 1] if "label_class" in df.columns else pd.DataFrame()
                imp = df[df["label_class"] == 0] if "label_class" in df.columns else pd.DataFrame()
                if len(kn):
                    summarize_object(kn, f"{split} KN")
                if len(imp):
                    summarize_object(imp, f"{split} Imposters")
        else:
            print(f"\n(no {p})")

    # 2. Raw imposter folders
    print("\n" + "=" * 60)
    print("  RAW imposter folders under data/raw/")
    print("=" * 60)
    if RAW.exists():
        for folder in sorted(RAW.iterdir()):
            if not folder.is_dir():
                continue
            files = list(folder.rglob("*.parquet"))
            print(f"\n  {folder.name}: {len(files)} parquet files")
            if files:
                df = load_one(files[0])
                if df is not None:
                    summarize_object(df, f"sample raw/{folder.name}/{files[0].name}")
    else:
        print("  data/raw does not exist")

    # 3. New imposters (if any)
    new_dir = ROOT / "data" / "raw_new_imposters"
    print("\n" + "=" * 60)
    print("  NEW imposters (data/raw_new_imposters/)")
    print("=" * 60)
    if new_dir.exists():
        for folder in sorted(new_dir.iterdir()):
            if not folder.is_dir():
                continue
            files = list(folder.rglob("*.parquet"))
            print(f"\n  {folder.name}: {len(files)} files")
            if files:
                df = load_one(files[0])
                if df is not None:
                    summarize_object(df, f"sample new/{folder.name}/{files[0].name}")
    else:
        print("  (folder does not exist yet – run the ANTARES fetcher first)")

    # 4. Quick V2 POSSIS sample (for comparison)
    print("\n" + "=" * 60)
    print("  V2 POSSIS sample (for comparison)")
    print("=" * 60)
    if V2.exists():
        v2_files = list(V2.glob("*.parquet"))
        print(f"  {len(v2_files)} V2 files")
        if v2_files:
            df = load_one(v2_files[0])
            if df is not None:
                summarize_object(df, f"sample V2 {v2_files[0].name}")
    else:
        print("  (no V2 folder)")

    print("\nDONE")

if __name__ == "__main__":
    main()