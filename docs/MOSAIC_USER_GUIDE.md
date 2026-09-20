# MOSAIC User Guide

**MOSAIC** scores optical transients for **kilonova** probability by combining:

1. **Stream 1 — RTF** (light-curve encoder → 128-d latent)  
2. **Stream 2 — Spatial** (GW skymap + GLADE+ host → 6-d features)  
3. **Stream 3 — Fusion** (MLP → **P(KN)** ∈ [0, 1])

This guide is for anyone who clones the repo and wants to run **one alert**, a **folder of alerts**, or a **live Kafka stream**.

---

## 1. Repository layout (what must exist)

```text
MOSAIC/
├── src/
│   ├── model.py              # Fusion MLP
│   ├── vectorizer.py         # LC → 18-channel tensor
│   ├── features.py           # spatial dict → 6-vector
│   ├── gw_processor.py       # skymap I/O + evaluate
│   ├── spatial.py            # active GW cache + GLADE host
│   ├── run_mosaic.py         # MOSAICPipeline (loads all 3 streams)
│   ├── cli.py                # single / batch CLI
│   └── realtime_kafka.py     # Kafka streaming worker
│
├── models/
│   ├── rtf_encoder_latents.onnx      # required
│   └── mosaic_fusion_best.pt         # required
│
├── docs/Model W&B/RTF_MOSAIC/
│   └── mosaic_rtf_stats.json         # required (same stats as training)
│
└── data/processed/GLADE+/
    ├── glade_ckdtree.pkl             # required
    └── glade_healpix_nside128.pkl    # required
```

Optional (not needed for inference):

- `inject.py`, training notebooks, `train_fusion.py`, raw POSSIS, etc.

### Environment

```bash
conda activate mosaic   # or any env with the deps below
pip install torch onnxruntime pandas numpy pyarrow \
            astropy astropy-healpix healpy scikit-learn
# for streaming only:
pip install kafka-python
# optional HTTP helper (if you use the file-watch service):
pip install flask
```

From repo root, run CLI as:

```bash
python src/cli.py ...
# or
PYTHONPATH=src python -m cli ...
```

---

## 2. Check that everything is wired

```bash
python src/cli.py status
```

You should see **OK** for ONNX, stats, fusion weights, and both GLADE pickles. Fix any **MISSING** paths before scoring.

---

## 3. Single event (one light curve)

### 3.1 Light-curve file

Parquet or CSV with at least:

| Column     | Meaning                          |
|------------|----------------------------------|
| `jd`       | Julian date of each detection    |
| `fid`      | Filter id: 1=g, 2=r, 3=i         |
| `magpsf`   | PSF magnitude                    |
| `sigmapsf` | Magnitude uncertainty            |

Helpful optional columns: `diffmaglim`, `objectId`, `ra`, `dec`.

Minimum **3** detections (after sorting / band filter).

### 3.2 Score without a GW map

Optical-only spatial fallback (no coincidence):

```bash
python src/cli.py score \
  --lc path/to/ZTF_candidate.parquet \
  --ra 150.123 \
  --dec -5.456 \
  --jd 2460124.15 \
  --json-out score.json
```

If the file already has `ra` / `dec` columns, you can omit those flags.  
If `--jd` is omitted, the **last** `jd` on the light curve is used.

### 3.3 Score with a GW map (ToO)

Register the skymap **once** (same time system as alert `jd` — **prefer JD**):

```bash
python src/cli.py register-gw \
  --event S230627c \
  --fits /absolute/path/to/S230627c.fits \
  --t0 2460123.75
```

Then score candidates as above. If the transient falls in the **90% credible region** within ~14 days of `t0`, Stream 2 fills GW/host features; otherwise you still get a P(KN) from photometry + neutral spatial defaults.

### 3.4 Library API (same engine)

```python
from pathlib import Path
from vectorizer import RTFVectorizer
from run_mosaic import MOSAICPipeline
import pandas as pd

engine = MOSAICPipeline(
    onnx_stream1_path=Path("models/rtf_encoder_latents.onnx"),
    glade_tree_path=Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl"),
    glade_hpx_path=Path("~/MOSAIC/data/processed/GLADE+/glade_healpix_nside128.pkl"),
    fusion_weights_path=Path("models/mosaic_fusion_best.pt"),
)
engine.spatial_engine.register_gw_event(
    "S230627c", Path("/path/to/map.fits"), t0=2460123.75
)

df = pd.read_parquet("candidate.parquet")
vec = RTFVectorizer("docs/Model W&B/RTF_MOSAIC/mosaic_rtf_stats.json")
x, mask = vec.vectorize(df)
p_kn = engine.process_alert(x, ra=150.1, dec=-5.4, alert_time=2460124.15, pad_mask=mask)
print(p_kn)
```

---

## 4. Folder of events (batch)

```bash
python src/cli.py score-dir \
  --dir path/to/alerts_folder/ \
  --out scores.csv
```

- Each file: `*.parquet` or `*.csv` with the same photometry columns.  
- Coordinates: `ra`/`dec` **in the file**, or pass fixed `--ra` / `--dec` for the whole folder.  
- Optional sidecar `stem.json`: `{"ra", "dec", "jd", "objectId"}`.

Output CSV/parquet columns include `objectId`, `p_kn`, `has_gw`, etc.

---

## 5. Real-time stream (Kafka)

### 5.1 Topics

| Topic            | Direction | Content                |
|------------------|-----------|------------------------|
| `mosaic.gw`      | in        | Register skymaps       |
| `mosaic.alerts`  | in        | ZTF-like candidates    |
| `mosaic.scores`  | out       | P(KN) + spatial fields |

### 5.2 Start the worker

```bash
pip install kafka-python

python src/realtime_kafka.py run \
  --bootstrap localhost:9092 \
  --alerts-topic mosaic.alerts \
  --gw-topic mosaic.gw \
  --scores-topic mosaic.scores \
  --group mosaic-infer
```

Models load **once**. Scale by running more workers with the **same** `--group` (Kafka partitions).

### 5.3 Publish a GW event

```json
{
  "event_id": "S230627c",
  "t0": 2460123.75,
  "fits_path": "/data/gw/S230627c.fits"
}
```

(`fits_b64` is supported for small maps; prefer a path on shared disk.)

### 5.4 Publish an alert

```json
{
  "objectId": "ZTF23abcd",
  "ra": 150.12,
  "dec": -5.33,
  "jd": 2460124.15,
  "photometry": [
    {"jd": 2460123.90, "fid": 1, "magpsf": 19.2, "sigmapsf": 0.10, "diffmaglim": 20.5},
    {"jd": 2460124.00, "fid": 2, "magpsf": 19.4, "sigmapsf": 0.12, "diffmaglim": 20.4},
    {"jd": 2460124.10, "fid": 1, "magpsf": 19.7, "sigmapsf": 0.14, "diffmaglim": 20.3}
  ]
}
```

### 5.5 Consume scores

Each score message includes roughly:

`objectId`, `p_kn`, `n_points`, `ra`, `dec`, `jd`, `has_gw`, `gw_cred_level`, `dp_dv`, `S_gal`, `host_dist_mpc`, `host_K_mag`, `gw_event_id`, `latency_ms`, `ts_utc`.

### 5.6 End-to-end live path

```text
GraceDB / GCN  ──►  mosaic.gw     ──►  worker registers map
ZTF broker     ──►  mosaic.alerts ──►  worker scores candidate
                                      │
                                      ▼
                                 mosaic.scores  ──►  UI / DB / alerts
```

A ZTF alert does **not** need a GW to be scored. If a map is active and the sky/time match, `has_gw=true` and spatial features are informative.

---

## 6. Necessities checklist

| Item | Why |
|------|-----|
| `rtf_encoder_latents.onnx` | Stream 1 embedding |
| `mosaic_rtf_stats.json` | **Must** match the stats used when the encoder was trained/exported |
| `mosaic_fusion_best.pt` | Stream 3 weights |
| GLADE+ `glade_ckdtree.pkl` + `glade_healpix_nside128.pkl` | Host / spatial features |
| Python deps above | Runtime |
| Kafka broker + `kafka-python` | Only for streaming mode |
| Skymap FITS readable by the worker | Only when using GW coincidence |

**Time system:** use the **same** convention for GW `t0` and alert `jd` (Julian Date recommended). Mixing Unix seconds and JD breaks the 14-day temporal gate.

**Light curves:** ≥3 points; bands `fid` ∈ {1,2,3}; magnitudes in a realistic ZTF range.

---

## 7. Common failures

| Symptom | Likely cause |
|---------|----------------|
| `MISSING` on `status` | Weights or GLADE not downloaded / wrong path |
| ONNX shape / pad_mask error | Pass both `x` and `mask` from `RTFVectorizer`; do not add an extra batch dim |
| Always `has_gw=false` | No map registered, wrong `t0`/`jd` units, or source outside 90% CR |
| Extremely peaked P(KN) on fusion offline tests | Check label leakage / tiny test set; trust calibration on real streams |
| Kafka no messages | Wrong bootstrap, topic names, or consumer `group` stuck on old offsets (`--offset-reset earliest` for replay) |

---

## 8. Quick reference

```bash
# health
python src/cli.py status

# one candidate
python src/cli.py score --lc CAND.parquet --ra RA --dec DEC

# many candidates
python src/cli.py score-dir --dir ./alerts --out scores.csv

# live stream
python src/realtime_kafka.py run --bootstrap HOST:9092
```

For maintainers: export ONNX and stats with your training/export scripts; rebuild GLADE pickles with `glade_ckdtree.py` and `build_healpix_index.py` if the catalog is updated.
