from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve package root (repo layout: MOSAIC/src/...)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULTS = {
    "onnx": ROOT / "models" / "rtf_encoder_latents.onnx",
    "stats": ROOT / "docs" / "Model W&B" / "RTF_MOSAIC" / "mosaic_rtf_stats.json",
    "fusion": ROOT / "models" / "mosaic_fusion_best.pt",
    "glade_tree": Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl").expanduser(),
    "glade_hpx": Path("~/MOSAIC/data/processed/GLADE+/glade_healpix_nside128.pkl").expanduser(),
    "gw_cache": ROOT / "data" / "processed" / "active_gw.json",
}


def _load_engine(args):
    """Build MOSAICPipeline from CLI paths."""
    from run_mosaic import MOSAICPipeline

    onnx = Path(args.onnx).expanduser()
    fusion = Path(args.fusion).expanduser() if args.fusion else None
    tree = Path(args.glade_tree).expanduser()
    hpx = Path(args.glade_hpx).expanduser()

    for p, name in [(onnx, "ONNX"), (tree, "GLADE tree"), (hpx, "GLADE HEALPix")]:
        if not p.exists():
            raise SystemExit(f"Missing {name}: {p}\nRun setup / download models first.")

    engine = MOSAICPipeline(
        onnx_stream1_path=onnx,
        glade_tree_path=tree,
        glade_hpx_path=hpx,
        fusion_weights_path=fusion if fusion and fusion.exists() else None,
    )
    # restore GW cache if present
    cache = Path(args.gw_cache).expanduser()
    if cache.exists():
        _restore_gw_cache(engine, cache)
    return engine


def _restore_gw_cache(engine, cache_path: Path):
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return
    for ev in data.get("events", []):
        fits = Path(ev["fits"]).expanduser()
        if fits.exists():
            try:
                engine.spatial_engine.register_gw_event(
                    ev["event_id"], fits, float(ev["t0"])
                )
            except Exception as e:
                print(f"  warn: could not restore GW {ev['event_id']}: {e}", file=sys.stderr)


def _save_gw_cache(engine, cache_path: Path):
    events = []
    for eid, d in engine.spatial_engine.active_gw_events.items():
        # t0 stored; fits path may be unknown — keep eid + t0 only if no path
        events.append({"event_id": eid, "t0": d["t0"], "fits": d.get("fits_path", "")})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"events": events}, indent=2))


def _vectorize_lc(df: pd.DataFrame, stats_path: Path):
    from vectorizer import RTFVectorizer

    vec = RTFVectorizer(stats_path=str(stats_path))
    x, mask = vec.vectorize(df)
    return x, mask


def _score_one(engine, df, ra, dec, alert_jd, stats_path: Path) -> dict:
    """Returns dict with p_kn, spatial breakdown, n_points."""
    import onnxruntime as ort

    x, mask = _vectorize_lc(df, stats_path)

    # Stream 1
    sess = engine.ort_session
    in0 = sess.get_inputs()[0].name
    names = [i.name for i in sess.get_inputs()]
    ort_in = {in0: x.astype(np.float32)}
    if len(names) > 1:
        ort_in[names[1]] = mask.astype(bool)
    z = sess.run(None, ort_in)[0]  # (1, 128)

    # Stream 2 — alert_time as JD days → approximate unix-ish if needed;
    # spatial.py uses alert_time for temporal gate vs t0.
    # We pass JD; register-gw should use same time system (JD).
    spatial_dict = engine.spatial_engine.process_alert(
        float(ra), float(dec), float(alert_jd)
    )

    from features import format_spatial_features

    x_sp = format_spatial_features(spatial_dict)

    # Stream 3
    with torch.no_grad():
        p = engine.fusion_model(
            torch.from_numpy(z).float(),
            torch.from_numpy(x_sp).unsqueeze(0).float(),
        )
        # model already has Sigmoid
        p_kn = float(p.item())

    return {
        "p_kn": p_kn,
        "n_points": int(len(df)),
        "has_gw": bool(spatial_dict.get("has_gw_coincidence", False)),
        "gw_cred_level": float(spatial_dict.get("gw_cred_level", 1.0)),
        "dp_dv": float(spatial_dict.get("dp_dv", 0.0)),
        "S_gal": float(spatial_dict.get("S_gal", 0.0)),
        "host_dist_mpc": float(spatial_dict.get("best_host_dist_mpc", -1.0)),
        "host_K_mag": float(spatial_dict.get("best_host_K_mag", 99.0)),
    }


def cmd_status(args):
    print("MOSAIC status")
    print(f"  repo root : {ROOT}")
    for k, p in [
        ("onnx", Path(args.onnx)),
        ("stats", Path(args.stats)),
        ("fusion", Path(args.fusion)),
        ("glade_tree", Path(args.glade_tree)),
        ("glade_hpx", Path(args.glade_hpx)),
    ]:
        p = p.expanduser()
        print(f"  {k:12s}: {'OK' if p.exists() else 'MISSING'}  {p}")
    cache = Path(args.gw_cache).expanduser()
    if cache.exists():
        data = json.loads(cache.read_text())
        print(f"  active GW : {len(data.get('events', []))} (from {cache})")
    else:
        print("  active GW : none")


def cmd_register_gw(args):
    engine = _load_engine(args)
    fits = Path(args.fits).expanduser()
    if not fits.exists():
        raise SystemExit(f"FITS not found: {fits}")
    engine.spatial_engine.register_gw_event(args.event, fits, float(args.t0))
    # stash fits path for cache
    engine.spatial_engine.active_gw_events[args.event]["fits_path"] = str(fits)
    _save_gw_cache(engine, Path(args.gw_cache).expanduser())
    print(f"Registered GW event {args.event}  t0={args.t0}  map={fits.name}")
    print(f"Active events: {list(engine.spatial_engine.active_gw_events)}")


def cmd_score(args):
    engine = _load_engine(args)
    lc_path = Path(args.lc).expanduser()
    if not lc_path.exists():
        raise SystemExit(f"Light curve not found: {lc_path}")
    df = pd.read_parquet(lc_path) if lc_path.suffix == ".parquet" else pd.read_csv(lc_path)
    df = df.sort_values("jd") if "jd" in df.columns else df

    ra = args.ra
    dec = args.dec
    if ra is None and "ra" in df.columns:
        ra = float(df["ra"].iloc[0])
    if dec is None and "dec" in df.columns:
        dec = float(df["dec"].iloc[0])
    if ra is None or dec is None:
        raise SystemExit("Need --ra and --dec (or ra/dec columns on the LC file)")

    jd = args.jd
    if jd is None:
        jd = float(df["jd"].iloc[-1]) if "jd" in df.columns else 0.0

    stats = Path(args.stats).expanduser()
    if not stats.exists():
        raise SystemExit(f"Missing RTF stats JSON: {stats}")

    out = _score_one(engine, df, ra, dec, jd, stats)
    oid = df["objectId"].iloc[0] if "objectId" in df.columns else lc_path.stem

    print("=" * 56)
    print(f"  object     : {oid}")
    print(f"  n_points   : {out['n_points']}")
    print(f"  P(KN)      : {out['p_kn']:.4f}")
    print(f"  GW coinc.  : {out['has_gw']}")
    print(f"  cred_level : {out['gw_cred_level']:.4f}")
    print(f"  log dp_dv  : {np.log10(out['dp_dv'] + 1e-30):.3f}")
    print(f"  host dist  : {out['host_dist_mpc']:.1f} Mpc")
    print(f"  S_gal      : {out['S_gal']:.3e}")
    print("=" * 56)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"objectId": str(oid), **out}, indent=2))
        print(f"Wrote {args.json_out}")


def cmd_score_dir(args):
    engine = _load_engine(args)
    stats = Path(args.stats).expanduser()
    d = Path(args.dir).expanduser()
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {d}")

    rows = []
    for i, fp in enumerate(files):
        df = pd.read_parquet(fp)
        if "jd" in df.columns:
            df = df.sort_values("jd")
        ra = args.ra
        dec = args.dec
        if ra is None and "ra" in df.columns:
            ra = float(df["ra"].iloc[0])
        if dec is None and "dec" in df.columns:
            dec = float(df["dec"].iloc[0])
        if ra is None or dec is None:
            print(f"  skip {fp.name}: no ra/dec", file=sys.stderr)
            continue
        jd = float(df["jd"].iloc[-1]) if "jd" in df.columns else 0.0
        try:
            out = _score_one(engine, df, ra, dec, jd, stats)
        except Exception as e:
            print(f"  fail {fp.name}: {e}", file=sys.stderr)
            continue
        oid = df["objectId"].iloc[0] if "objectId" in df.columns else fp.stem
        rows.append({"objectId": oid, "file": str(fp), **out})
        if (i + 1) % 25 == 0:
            print(f"  scored {i+1}/{len(files)}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} scores → {out_path}")
    if len(out_df):
        print(out_df[["objectId", "p_kn", "has_gw"]].head(10).to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser(
        prog="mosaic",
        description="MOSAIC real-time kilonova classifier (RTF × GW/GLADE × fusion)",
    )
    p.add_argument("--onnx", default=str(DEFAULTS["onnx"]))
    p.add_argument("--stats", default=str(DEFAULTS["stats"]))
    p.add_argument("--fusion", default=str(DEFAULTS["fusion"]))
    p.add_argument("--glade-tree", default=str(DEFAULTS["glade_tree"]))
    p.add_argument("--glade-hpx", default=str(DEFAULTS["glade_hpx"]))
    p.add_argument("--gw-cache", default=str(DEFAULTS["gw_cache"]))

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Check models and GW cache")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("register-gw", help="Register a GW skymap for spatial matching")
    s.add_argument("--event", required=True, help="Event id, e.g. S230627c")
    s.add_argument("--fits", required=True, help="Path to bayestar/skymap FITS")
    s.add_argument("--t0", type=float, required=True, help="Merger time (use JD to match --jd)")
    s.set_defaults(func=cmd_register_gw)

    s = sub.add_parser("score", help="Score one light curve")
    s.add_argument("--lc", required=True, help="Parquet/CSV with jd,fid,magpsf,sigmapsf,...")
    s.add_argument("--ra", type=float, default=None)
    s.add_argument("--dec", type=float, default=None)
    s.add_argument("--jd", type=float, default=None, help="Alert time (default: last jd on LC)")
    s.add_argument("--json-out", default=None)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("score-dir", help="Score all parquets in a directory")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", default="scores.csv")
    s.add_argument("--ra", type=float, default=None, help="Fixed RA if not in files")
    s.add_argument("--dec", type=float, default=None)
    s.set_defaults(func=cmd_score_dir)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
