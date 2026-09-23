import os
import json
import signal
import logging
import requests
import threading
from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from gcn_kafka import Consumer as GCNConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure local storage directory exists for downloaded GW skymaps
SKYMAP_DIR = Path("./data/skymaps")
SKYMAP_DIR.mkdir(parents=True, exist_ok=True)

# Shared shutdown flag across threads
stop_event = threading.Event()

local_producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def download_skymap(skymap_url: str, event_id: str) -> str:
    """Downloads FITS skymap from GCN/LVK URL to local disk for HEALPix processing."""
    if not skymap_url:
        return ""
    
    local_path = SKYMAP_DIR / f"{event_id}_skymap.fits"
    if local_path.exists():
        return str(local_path)

    try:
        logging.info(f"Downloading skymap for {event_id} from {skymap_url}...")
        response = requests.get(skymap_url, timeout=15)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Saved skymap to {local_path}")
        return str(local_path)
    except Exception as e:
        logging.error(f"Failed to download skymap for {event_id}: {e}")
        return ""

def ingest_gw_stream():
    """Consumes LVK Gravitational Wave alerts from GCN and pipes to local MOSAIC."""
    logging.info("Starting GW ingestion thread...")
    
    gcn_consumer = GCNConsumer(
        client_id='7l4p18l0p3g2r9jn9ctsel6eqi',
        client_secret='1a6avq96dicp3bpk410kdoje1mett9h9r5krtban24hrgirbvk42',
        config={'broker.address.family': 'v4', 'auto.offset.reset': 'latest'}
    )
    gcn_consumer.subscribe(['igwn.gwalert'])
    
    while not stop_event.is_set():
        for message in gcn_consumer.consume(timeout=1.0):
            if message.error():
                logging.error(f"GCN Error: {message.error()}")
                continue
                
            try:
                payload = json.loads(message.value().decode('utf-8'))
                event_id = payload.get("superevent_id", "UNKNOWN")
                skymap_url = payload.get("urls", {}).get("skymap")
                
                # Fetch FITS file to local disk before notifying downstream workers
                fits_path = download_skymap(skymap_url, event_id) if skymap_url else ""

                mosaic_gw_msg = {
                    "event_id": event_id,
                    "t0": payload.get("event", {}).get("time"),
                    "fits_path": fits_path,
                    "alert_type": payload.get("alert_type")
                }
                
                local_producer.send('mosaic.gw', mosaic_gw_msg)
                local_producer.flush()
                logging.info(f"Ingested GW Event: {event_id} | Skymap: {fits_path}")
            except Exception as e:
                logging.error(f"Error parsing GW payload: {e}")

    gcn_consumer.close()

def ingest_ztf_stream():
    """Consumes optical alerts from ZTF broker stream and pipes to local MOSAIC."""
    logging.info("Starting ZTF ingestion thread...")
    
    ztf_consumer = KafkaConsumer(
        'ztf_public_alerts',
        bootstrap_servers=['ztf-stream.broker.edu:9092'], 
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        consumer_timeout_ms=1000
    )
    
    while not stop_event.is_set():
        try:
            for message in ztf_consumer:
                if stop_event.is_set():
                    break
                alert = message.value
                candidate = alert.get("candidate", {})
                
                mosaic_alert_msg = {
                     "objectId": alert.get("objectId"),
                     "ra": candidate.get("ra"),
                     "dec": candidate.get("dec"),
                     "jd": candidate.get("jd"),
                     "photometry": alert.get("prv_candidates", []) + [candidate]
                }
                local_producer.send('mosaic.alerts', mosaic_alert_msg)
                logging.info(f"Ingested ZTF Alert: {mosaic_alert_msg['objectId']}")
        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"ZTF stream consumption error: {e}")

    ztf_consumer.close()

if __name__ == "__main__":
    gw_thread = threading.Thread(target=ingest_gw_stream, daemon=True)
    ztf_thread = threading.Thread(target=ingest_ztf_stream, daemon=True)
    
    gw_thread.start()
    ztf_thread.start()

    def handle_shutdown(sig, frame):
        logging.info("Shutdown requested. Stopping stream threads...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    gw_thread.join()
    ztf_thread.join()
    logging.info("Stream ingestion cleanly terminated.")