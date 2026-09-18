#!/usr/bin/env python3
"""ANTARES → MOSAIC 14-col parquet (real tags only)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from antares_client.search import search

ROOT = Path("~/MOSAIC").expanduser()
OUT = ROOT / "data" / "raw_new_imposters"
OUT.mkdir(parents=True, exist_ok=True)

ZTF_ZP = 23.9
MIN_DET = 3
MAX_PER = 1250
SLEEP = 0.12

# real tags from get_available_tags()
TARGET = {
    "AGN": ["nuclear_transient"],
    "RR Lyrae": ["high_amplitude_variable_star_candidate"],
    "Supernova": ["SN_candies", "young_extragalactic_candidate", "high_amplitude_transient_candidate"],
    "CV": ["dwarf_nova_outburst"],
    "Transient": ["blue_transient", "high_amplitude_transient_candidate"],  # extra bucket
}

SCHEMA = [
    "objectId", "group_id", "jd", "time_day", "fid",
    "magpsf", "sigmapsf", "flux", "flux_err", "diffmaglim",
    "label_class", "subclass", "dist_mpc", "angle_idx",
]


def fid_from_band(b) -> int:
    s = str(b).lower()
    if "g" in s or s in ("1", "zg"):
        return 1
    if "r" in s or s in ("2", "zr"):
        return 2
    if "i" in s or s in ("3", "zi"):
        return 3
    return 0


def locus_to_df(locus, subclass: str) -> pd.DataFrame | None:
    lc = getattr(locus, "lightcurve", None)
    if lc is None or len(lc) == 0:
        return None
    df = lc.copy()
    cols = {c.lower(): c for c in df.columns}

    def C(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_mjd = C("ant_mjd", "mjd")
    c_mag = C("ant_mag", "mag", "magpsf")
    c_err = C("ant_magerr", "magerr", "sigmapsf")
    c_band = C("ant_passband", "passband", "band", "filter")
    if not c_mjd or not c_mag:
        return None

    mjd = pd.to_numeric(df[c_mjd], errors="coerce").to_numpy()
    mag = pd.to_numeric(df[c_mag], errors="coerce").to_numpy()
    err = (
        pd.to_numeric(df[c_err], errors="coerce").to_numpy()
        if c_err is not None
        else np.full(len(mag), 0.1, dtype=float)
    )
    if c_band is not None:
        fid = np.array([fid_from_band(x) for x in df[c_band]], dtype=np.int8)
    else:
        fid = np.ones(len(mag), dtype=np.int8)

    ok = np.isfinite(mjd) & np.isfinite(mag) & (mag > 12) & (mag < 22.5)
    ok &= np.isfinite(err) & (err > 0) & (err < 1.5) & (fid > 0)
    if ok.sum() < MIN_DET:
        return None

    mjd, mag, err, fid = mjd[ok], mag[ok], err[ok], fid[ok]
    order = np.argsort(mjd)
    mjd, mag, err, fid = mjd[order], mag[order], err[order], fid[order]

    # prefer ZTF id from properties if present
    oid = str(locus.locus_id)
    props = getattr(locus, "properties", None) or {}
    for k, v in props.items():
        if isinstance(v, str) and v.startswith("ZTF"):
            oid = v
            break

    t0 = float(mjd.min())
    flux = 10 ** (-0.4 * (mag - ZTF_ZP))
    flux_err = np.maximum(flux * err * (0.4 * np.log(10)), 1e-12)

    out = pd.DataFrame(
        {
            "objectId": oid,
            "group_id": oid,
            "jd": (mjd + 2400000.5).astype(np.float64),
            "time_day": (mjd - t0).astype(np.float32),
            "fid": fid.astype(np.int8),
            "magpsf": mag.astype(np.float32),
            "sigmapsf": err.astype(np.float32),
            "flux": flux.astype(np.float32),
            "flux_err": flux_err.astype(np.float32),
            "diffmaglim": np.full(len(mag), 20.5, dtype=np.float32),
            "label_class": np.int64(0),
            "subclass": subclass,
            "dist_mpc": np.float32(np.nan),
            "angle_idx": np.int8(0),
        }
    )
    return out[SCHEMA]


def fetch_one(subclass: str, tags: list[str], limit: int) -> int:
    folder = OUT / subclass.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    saved, seen = 0, set()

    for tag in tags:
        if saved >= limit:
            break
        print(f"  [{subclass}] tag={tag!r}")
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"tags": tag}},
                        {
                            "range": {
                                "properties.num_mag_values": {
                                    "gte": MIN_DET,
                                    "lte": 300,
                                }
                            }
                        },
                    ]
                }
            }
        }
        try:
            for locus in search(query):
                if saved >= limit:
                    break
                lid = str(locus.locus_id)
                if lid in seen:
                    continue
                seen.add(lid)
                df = locus_to_df(locus, subclass)
                if df is None:
                    continue
                path = folder / f"{df['objectId'].iloc[0]}.parquet"
                if path.exists():
                    continue
                df.to_parquet(path, index=False)
                saved += 1
                if saved % 20 == 0:
                    print(f"    saved {saved}/{limit}")
                time.sleep(SLEEP)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}")
    return saved


def main():
    print("=" * 60)
    print("  ANTARES real-tag ingest →", OUT)
    print("=" * 60)
    for sub, tags in TARGET.items():
        n = fetch_one(sub, tags, MAX_PER)
        print(f"  → {sub}: {n} objects\n")
    print("Done.")


if __name__ == "__main__":
    main()