#!/usr/bin/env python3
"""
Fink Broker Imposter Fetcher for MOSAIC
=======================================
Queries the Fink REST API for fresh ZTF imposters (RR Lyrae, AGN, Supernovae)
independent of ALérce, and formats them into MOSAIC's parquet schema.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUTPUT_DIR = Path("~/MOSAIC/data/raw_new_imposters").expanduser()

# Fink classification mapping to MOSAIC folder structure
FINK_CLASSES = {
    "RR_Lyrae": "RRLyr",
    "AGN": "AGN",
    "Supernova": "SN candidate"
}

SCHEMA_COLUMNS = [
    "objectId", "group_id", "jd", "time_day", "fid",
    "magpsf", "sigmapsf", "flux", "flux_err", "diffmaglim",
    "label_class", "subclass", "dist_mpc", "angle_idx",
]

def fetch_class_from_fink(fink_class: str, limit: int = 50):
    url = "https://api.fink-portal.org/v1/search"
    payload = {
        "class": fink_class,
        "nlim": limit,
        "output-format": "json"
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"  [ERROR] Failed to fetch {fink_class}: {response.text}")
        return []
    
    data = response.json()
    if not data or "data" not in data:
        return []
    
    df_raw = pd.DataFrame(data["data"])
    if df_raw.empty:
        return []
    
    # Fink returns multi-alert entries grouped by ijd / djd or objectId
    object_groups = df_raw.groupby("i:objectId") if "i:objectId" in df_raw.columns else df_raw.groupby("objectId")
    extracted_dfs = []

    for obj_id, group in object_groups:
        if len(group) < 3:
            continue
        
        # Map Fink columns to MOSAIC schema
        jd = group["i:jd"].values if "i:jd" in group.columns else group["jd"].values
        t_day = jd - jd.min()
        
        formatted = pd.DataFrame({
            "objectId": str(obj_id),
            "group_id": f"{fink_class}__{obj_id}",
            "jd": jd,
            "time_day": t_day,
            "fid": group["i:fid"].values if "i:fid" in group.columns else group["fid"].values,
            "magpsf": group["i:magpsf"].values if "i:magpsf" in group.columns else group["magpsf"].values,
            "sigmapsf": group["i:sigmapsf"].values if "i:sigmapsf" in group.columns else group["sigmapsf"].values,
            "flux": group["i:flux"].values if "i:flux" in group.columns else group.get("flux", np.zeros(len(group))),
            "flux_err": group["i:fluxerr"].values if "i:fluxerr" in group.columns else group.get("fluxerr", np.ones(len(group))),
            "diffmaglim": group["i:diffmaglim"].values if "i:diffmaglim" in group.columns else group["diffmaglim"].values,
            "label_class": 0,
            "subclass": fink_class,
            "dist_mpc": np.float32(np.nan),
            "angle_idx": np.int8(-1),
        })
        
        extracted_dfs.append((str(obj_id), formatted[SCHEMA_COLUMNS]))
        
    return extracted_dfs

def main():
    print("=" * 60)
    print("  Fetching Fresh Imposters from Fink Broker API")
    print("=" * 60)
    
    for sub_folder, fink_query_name in FINK_CLASSES.items():
        target_dir = OUTPUT_DIR / sub_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Querying Fink for class: {fink_query_name}...")
        results = fetch_class_from_fink(fink_query_name, limit=100)
        print(f"  -> Retrieved {len(results)} unique objects for {sub_folder}.")
        
        for obj_id, df_obj in results:
            out_path = target_dir / f"{obj_id}.parquet"
            df_obj.to_parquet(out_path, index=False)
            
    print(f"\nFresh imposters successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()