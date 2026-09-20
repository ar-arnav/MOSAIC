from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

log = logging.getLogger("mosaic.kafka")


def setup_log(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


DEFAULTS = {
    "onnx": ROOT / "models" / "rtf_encoder_latents.onnx",
    "stats": ROOT / "docs" / "Model W&B" / "RTF_MOSAIC" / "mosaic_rtf_stats.json",
    "fusion": ROOT / "models" / "mosaic_fusion_best.pt",
    "glade_tree": Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl").expanduser(),
    "glade_hpx": Path("~/MOSAIC/data/processed/GLADE+/glade_healpix_nside128.pkl").expanduser(),
    "bootstrap": "localhost:9092",
    "alerts_topic": "mosaic.alerts",
    "gw_topic": "mosaic.gw",
    "scores_topic": "mosaic.scores",
    "group": "mosaic-infer",
}


class MosaicKafkaWorker:
    def __init__(self, args):
        from run_mosaic import MOSAICPipeline
        from vectorizer import RTFVectorizer
        from features import format_spatial_features
        from kafka import KafkaConsumer, KafkaProducer

        self.format_spatial = format_spatial_features
        self.min_points = int(args.min_points)
        self.stats_path = Path(args.stats).expanduser()
        self.tmp_gw_dir = Path(args.tmp_gw_dir).expanduser()
        self.tmp_gw_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading models (once)...")
        self.engine = MOSAICPipeline(
            onnx_stream1_path=Path(args.onnx).expanduser(),
            glade_tree_path=Path(args.glade_tree).expanduser(),
            glade_hpx_path=Path(args.glade_hpx).expanduser(),
            fusion_weights_path=Path(args.fusion).expanduser(),
        )
        self.vectorizer = RTFVectorizer(stats_path=str(self.stats_path))

        common = dict(
            bootstrap_servers=args.bootstrap.split(","),
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        self.consumer = KafkaConsumer(
            args.alerts_topic,
            args.gw_topic,
            group_id=args.group,
            auto_offset_reset=args.offset_reset,
            enable_auto_commit=True,
            consumer_timeout_ms=1000,  # allow loop wakeups
            **common,
        )
        # map topic → handler via subscription list order not needed; check msg.topic
        self.alerts_topic = args.alerts_topic
        self.gw_topic = args.gw_topic
        self.scores_topic = args.scores_topic

        self.producer = KafkaProducer(
            bootstrap_servers=args.bootstrap.split(","),
            value_serializer=lambda d: json.dumps(d).encode("utf-8"),
            key_serializer=lambda k: (k or "").encode("utf-8"),
            acks="all",
            retries=3,
        )
        log.info(
            "Kafka connected  bootstrap=%s  in=[%s,%s]  out=%s  group=%s",
            args.bootstrap,
            args.alerts_topic,
            args.gw_topic,
            args.scores_topic,
            args.group,
        )

    # ---------------------------------------------------------------- GW
    def handle_gw(self, msg: dict) -> None:
        event_id = msg["event_id"]
        t0 = float(msg["t0"])
        if "fits_path" in msg and msg["fits_path"]:
            fits = Path(msg["fits_path"]).expanduser()
            if not fits.exists():
                log.error("GW fits_path missing on disk: %s", fits)
                return
        elif "fits_b64" in msg:
            raw = base64.b64decode(msg["fits_b64"])
            fits = self.tmp_gw_dir / f"{event_id}.fits"
            fits.write_bytes(raw)
        else:
            log.error("GW message needs fits_path or fits_b64: %s", event_id)
            return

        self.engine.spatial_engine.register_gw_event(event_id, fits, t0)
        self.engine.spatial_engine.active_gw_events[event_id]["fits_path"] = str(fits)
        log.info("GW registered from stream: %s t0=%s", event_id, t0)

    # ---------------------------------------------------------------- alert
    def handle_alert(self, msg: dict) -> dict | None:
        t_wall0 = time.perf_counter()
        oid = str(msg.get("objectId", "unknown"))
        ra = float(msg["ra"])
        dec = float(msg["dec"])
        phot = msg.get("photometry") or msg.get("detections")
        if not phot:
            log.warning("Alert %s has no photometry", oid)
            return None
        df = pd.DataFrame(phot)
        if "jd" not in df.columns:
            log.warning("Alert %s photometry missing jd", oid)
            return None
        df = df.sort_values("jd")
        if len(df) < self.min_points:
            log.warning("Alert %s too short (%d)", oid, len(df))
            return None

        alert_time = float(msg.get("jd", df["jd"].iloc[-1]))

        # Stream 1
        x, mask = self.vectorizer.vectorize(df)
        sess = self.engine.ort_session
        names = [i.name for i in sess.get_inputs()]
        ort_in = {names[0]: x.astype(np.float32)}
        if len(names) > 1:
            ort_in[names[1]] = mask.astype(bool)
        z = sess.run(None, ort_in)[0]

        # Stream 2
        spatial = self.engine.spatial_engine.process_alert(ra, dec, alert_time)
        x_sp = self.format_spatial(spatial)

        # Stream 3
        import torch

        with torch.no_grad():
            p = self.engine.fusion_model(
                torch.from_numpy(z).float(),
                torch.from_numpy(x_sp).unsqueeze(0).float(),
            )
            p_kn = float(p.item())

        latency_ms = (time.perf_counter() - t_wall0) * 1000.0
        out = {
            "objectId": oid,
            "p_kn": p_kn,
            "n_points": int(len(df)),
            "ra": ra,
            "dec": dec,
            "jd": alert_time,
            "has_gw": bool(spatial.get("has_gw_coincidence", False)),
            "gw_cred_level": float(spatial.get("gw_cred_level", 1.0)),
            "dp_dv": float(spatial.get("dp_dv", 0.0)),
            "S_gal": float(spatial.get("S_gal", 0.0)),
            "host_dist_mpc": float(spatial.get("best_host_dist_mpc", -1.0)),
            "host_K_mag": float(spatial.get("best_host_K_mag", 99.0)),
            "gw_event_id": spatial.get("gw_event_id"),
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency_ms, 2),
        }
        return out

    def publish_score(self, score: dict) -> None:
        key = score.get("objectId", "")
        fut = self.producer.send(self.scores_topic, key=key, value=score)
        fut.get(timeout=10)
        log.info(
            "SCORED %s  P(KN)=%.4f  GW=%s  latency=%.0fms",
            score["objectId"],
            score["p_kn"],
            score["has_gw"],
            score["latency_ms"],
        )

    def run_forever(self):
        log.info("Consuming streams… Ctrl+C to stop")
        try:
            while True:
                polled = self.consumer.poll(timeout_ms=1000)
                if not polled:
                    # optional: expire GW by wall clock if t0 was unix
                    continue
                for tp, records in polled.items():
                    for rec in records:
                        topic = rec.topic
                        try:
                            if topic == self.gw_topic:
                                self.handle_gw(rec.value)
                            elif topic == self.alerts_topic:
                                score = self.handle_alert(rec.value)
                                if score is not None:
                                    self.publish_score(score)
                            else:
                                log.debug("Ignore topic %s", topic)
                        except Exception:
                            log.exception(
                                "Failed message topic=%s key=%s",
                                topic,
                                rec.key,
                            )
        except KeyboardInterrupt:
            log.info("Shutting down")
        finally:
            self.consumer.close()
            self.producer.flush()
            self.producer.close()


def build_parser():
    p = argparse.ArgumentParser(description="MOSAIC Kafka streaming inference")
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--bootstrap", default=DEFAULTS["bootstrap"])
    p.add_argument("--alerts-topic", default=DEFAULTS["alerts_topic"])
    p.add_argument("--gw-topic", default=DEFAULTS["gw_topic"])
    p.add_argument("--scores-topic", default=DEFAULTS["scores_topic"])
    p.add_argument("--group", default=DEFAULTS["group"])
    p.add_argument(
        "--offset-reset",
        default="latest",
        choices=["latest", "earliest"],
    )
    p.add_argument("--onnx", default=str(DEFAULTS["onnx"]))
    p.add_argument("--stats", default=str(DEFAULTS["stats"]))
    p.add_argument("--fusion", default=str(DEFAULTS["fusion"]))
    p.add_argument("--glade-tree", default=str(DEFAULTS["glade_tree"]))
    p.add_argument("--glade-hpx", default=str(DEFAULTS["glade_hpx"]))
    p.add_argument("--tmp-gw-dir", default=str(ROOT / "data" / "realtime" / "gw_tmp"))
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_log(args.verbose)
    if args.cmd != "run":
        raise SystemExit("use: run")
    try:
        from kafka import KafkaConsumer  # noqa: F401
    except ImportError:
        raise SystemExit("pip install kafka-python")
    worker = MosaicKafkaWorker(args)
    worker.run_forever()


if __name__ == "__main__":
    main()
