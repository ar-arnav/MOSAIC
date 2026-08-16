#!/usr/bin/env python3
"""
MOSAIC ZTF Imposter Downloader
- Process one class fully before moving to the next
- Multiple workers share the same class
- Save one parquet per object: <objectId>.parquet
- Verbose progress: downloading / converted / failed
"""

import os
import sys
import time
import json
import logging
import argparse
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alerce.core import Alerce


def setup_logging(worker_id=None):
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler("imposters_download.log")
    fh.setFormatter(fmt)
    root.setLevel(logging.INFO)
    root.addHandler(sh)
    root.addHandler(fh)


setup_logging()
log = logging.getLogger("imposters")


# Targets (hard negatives intentionally boosted)
FULL_TAXONOMY = {
    # "SNIa": 12_000,
    # "SNII": 6_000,
    # "SNIbc": 2_500,
    # "SLSN": 500,
    # "CV/Nova": 6_000,
    # "AGN": 15_000,
    # "QSO": 4_000,
    # "Blazar": 2_000,
    # "LPV": 4_000,
    # "E": 15_000,
    # "RRL": 8_000,
    # "YSO": 1_500,
    # "SNIIb": 1_000,
    'TDE': 100,
    'DSCT': 1_000,
    'Periodic-Other': 1_000,
}


PARQUET_SCHEMA = pa.schema([
    ("objectId", pa.string()),
    ("label_class", pa.string()),
    ("angle_idx", pa.int8()),
    ("jd", pa.float64()),
    ("time_day", pa.float32()),
    ("fid", pa.int8()),
    ("magpsf", pa.float32()),
    ("sigmapsf", pa.float32()),
    ("flux", pa.float32()),
    ("flux_err", pa.float32()),
    ("diffmaglim", pa.float32()),
])


def mag_to_flux(mag, sigmamag):
    flux = 10.0 ** (-0.4 * (mag - 23.9))
    flux_err = flux * (sigmamag * (np.log(10.0) / 2.5))
    return flux.astype(np.float32), flux_err.astype(np.float32)


# ---------------------------------------------------------------------------
# Catalog harvesting (single process – gets the OID list for one class)
# ---------------------------------------------------------------------------
def harvest_oids(class_name, target_count, min_prob=0.4, page_size=100):
    """Return up to target_count unique OIDs for this class."""
    alerce = Alerce()
    oids = []
    seen = set()
    page = 1
    empty_streak = 0

    log.info(f"--- Harvesting OIDs for {class_name} (target={target_count}, min_prob={min_prob}) ---")

    while len(oids) < target_count:
        try:
            objs = alerce.query_objects(
                survey="ztf",
                classifier="lc_classifier",
                class_name=class_name,
                ranking=1,
                probability=min_prob,
                order_by="probability",
                order_mode="DESC",
                page=page,
                page_size=page_size,
                format="pandas",
            )
        except Exception as e:
            log.warning(f"[{class_name}] page {page} error: {e}")
            time.sleep(2)
            page += 1
            empty_streak += 1
            if empty_streak >= 15:
                break
            continue

        if objs is None or objs.empty:
            empty_streak += 1
            if empty_streak >= 15:
                log.warning(f"[{class_name}] catalog exhausted after {page} pages")
                break
            page += 1
            continue

        empty_streak = 0

        if "oid" in objs.columns:
            raw = objs["oid"].astype(str).tolist()
        elif objs.index.name == "oid":
            raw = objs.index.astype(str).tolist()
        else:
            tmp = objs.reset_index()
            raw = tmp["oid"].astype(str).tolist() if "oid" in tmp.columns else []

        for oid in raw:
            if oid not in seen:
                seen.add(oid)
                oids.append(oid)
                if len(oids) >= target_count:
                    break

        log.info(f"[{class_name}] page {page}: collected {len(oids)}/{target_count} OIDs")
        page += 1
        time.sleep(0.3)

    log.info(f"[{class_name}] Harvest finished → {len(oids)} unique OIDs")
    return oids


# ---------------------------------------------------------------------------
# Single-object download + convert (runs inside a worker process)
# ---------------------------------------------------------------------------
def process_one_object(args):
    """
    args = (oid, class_name, out_dir)
    Returns a short status string.
    """
    oid, class_name, out_dir = args
    out_path = Path(out_dir) / f"{oid}.parquet"

    if out_path.exists():
        return f"SKIP already exists: {oid}"

    try:
        alerce = Alerce()
        dets = alerce.query_detections(oid=oid, survey="ztf", format="pandas")
        time.sleep(0.05)

        if dets is None or dets.empty:
            return f"FAIL empty detections: {oid}"

        if "oid" not in dets.columns:
            dets = dets.copy()
            dets["oid"] = oid

        dets = dets[dets["fid"].isin([1, 2, 3])].copy()
        dets = dets.dropna(subset=["magpsf", "sigmapsf", "mjd"])

        if len(dets) < 2:
            return f"FAIL <2 good points: {oid}"

        df = pd.DataFrame()
        df["objectId"] = dets["oid"].astype(str)
        df["label_class"] = class_name
        df["angle_idx"] = np.int8(-1)

        mjd = dets["mjd"].astype(np.float64)
        df["jd"] = mjd + 2400000.5
        df["time_day"] = (
            dets.groupby("oid")["mjd"]
            .transform(lambda x: x - x.min())
            .astype(np.float32)
        )
        df["fid"] = dets["fid"].astype(np.int8)
        df["magpsf"] = dets["magpsf"].astype(np.float32)
        df["sigmapsf"] = dets["sigmapsf"].astype(np.float32)

        flux, flux_err = mag_to_flux(df["magpsf"].to_numpy(), df["sigmapsf"].to_numpy())
        df["flux"] = flux
        df["flux_err"] = flux_err

        if "diffmaglim" in dets.columns:
            df["diffmaglim"] = dets["diffmaglim"].fillna(20.5).astype(np.float32)
        else:
            df["diffmaglim"] = np.float32(20.5)

        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, out_path, compression="SNAPPY")

        return f"OK  {oid}  ({len(df)} rows)"

    except Exception as e:
        return f"FAIL {oid}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Process one full class with a pool of workers
# ---------------------------------------------------------------------------
def process_class(class_name, target_count, num_workers=3, min_prob=0.4):
    class_dir = Path(os.path.expanduser(f"~/MOSAIC/data/raw/{class_name}"))
    class_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip objects that already have a parquet
    existing = {p.stem for p in class_dir.glob("*.parquet")}
    log.info(f"[{class_name}] {len(existing)} objects already on disk")

    oids = harvest_oids(class_name, target_count, min_prob=min_prob)
    remaining = [oid for oid in oids if oid not in existing]

    if not remaining:
        log.info(f"[{class_name}] Nothing left to download. Done.")
        return

    log.info(
        f"[{class_name}] Need to download {len(remaining)} objects "
        f"with {num_workers} workers"
    )

    tasks = [(oid, class_name, str(class_dir)) for oid in remaining]

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(process_one_object, t): t[0] for t in tasks}

        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if status.startswith("OK"):
                ok += 1
            elif status.startswith("SKIP"):
                skip += 1
            else:
                fail += 1

            # print every object
            log.info(f"[{class_name}] [{i}/{len(remaining)}] {status}")

            if i % 50 == 0 or i == len(remaining):
                log.info(
                    f"[{class_name}] Summary so far → "
                    f"OK={ok}  FAIL={fail}  SKIP={skip}"
                )

    log.info(
        f"[{class_name}] FINISHED  OK={ok}  FAIL={fail}  SKIP={skip}  "
        f"Total on disk ≈ {len(list(class_dir.glob('*.parquet')))}"
    )


# ---------------------------------------------------------------------------
# Main – classes one after another
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Only SNIa, target=30")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-prob", type=float, default=0.4)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of class names, e.g. --classes SNIa E RRL",
    )
    args = parser.parse_args()

    if args.test:
        log.info("TEST MODE: SNIa target=30")
        process_class("SNIa", 30, num_workers=args.workers, min_prob=args.min_prob)
        return

    taxonomy = FULL_TAXONOMY
    if args.classes:
        taxonomy = {k: v for k, v in FULL_TAXONOMY.items() if k in args.classes}

    log.info(f"Will process classes in order: {list(taxonomy.keys())}")
    log.info(f"Workers per class: {args.workers}")

    for class_name, target in taxonomy.items():
        log.info("=" * 70)
        log.info(f"STARTING CLASS: {class_name}  (target {target})")
        log.info("=" * 70)
        process_class(
            class_name,
            target,
            num_workers=args.workers,
            min_prob=args.min_prob,
        )

    log.info("ALL CLASSES COMPLETE")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()