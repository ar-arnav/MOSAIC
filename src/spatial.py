import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import astropy_healpix as ah
from astropy import units as u

from gw_processor import GW_data


class SpatialPipeline:
    """
    MOSAIC Spatial Cross-Matching Module.
    Handles GLADE+ host galaxy indexing and real-time GW 3D volume matching.
    """

    def __init__(self, glade_ckdtree_path: Path, glade_healpix_path: Path, target_nside: int = 128):
        self.target_nside = target_nside
        
        # Load spatial index catalogs
        print("Loading GLADE+ spatial indices...")
        self.glade_dict = self._load_pkl(Path(glade_ckdtree_path).expanduser()) #[cite: 2, 3]
        self.pixel_map = self._load_pkl(Path(glade_healpix_path).expanduser()) #[cite: 1, 2]
        
        # Active GW event cache: {event_id: {"gw_obj": GW_data, "t0": float}}
        self.active_gw_events = {}

    @staticmethod
    def _load_pkl(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def register_gw_event(self, event_id: str, fits_path: Path, t0: float):
        """Registers a new LIGO/Virgo/KAGRA alert into the active memory cache.""" #[cite: 4]
        print(f"Registering GW event {event_id} (t0={t0})...")
        gw_obj = GW_data(Path(fits_path).expanduser(), t0=t0) #[cite: 4]
        self.active_gw_events[event_id] = {"gw_obj": gw_obj, "t0": t0}

    def remove_expired_events(self, current_time: float, max_age_days: float = 14.0):
        """Purges GW events older than the cutoff threshold (default 14 days)."""
        max_age_sec = max_age_days * 86400.0
        expired = [
            eid for eid, data in self.active_gw_events.items()
            if (current_time - data["t0"]) > max_age_sec
        ]
        for eid in expired:
            del self.active_gw_events[eid]
            print(f"Purged expired GW event: {eid}")

    def get_fallback_features(self) -> dict:
        """Returns neutral baseline features when no GW event triggers."""
        return {
            "has_gw_coincidence": False,
            "gw_cred_level": 1.0,
            "dp_dv": 0.0,
            "best_host_dist_mpc": -1.0,
            "best_host_K_mag": 99.0,
            "S_gal": 0.0,
        }

    def process_alert(
        self, ra_deg: float, dec_deg: float, alert_time: float, host_dl: float = None
    ) -> dict:
        """
        Processes a single incoming ZTF alert.
        Checks active GW triggers and computes spatial/host features.
        """
        if not self.active_gw_events:
            return self.get_fallback_features()

        best_match = None
        highest_dp_dv = -1.0

        for event_id, data in self.active_gw_events.items():
            gw = data["gw_obj"] #[cite: 4]
            t0 = data["t0"]

            # Temporal Gate: Optical alert must occur within [0, 14] days after GW merger
            dt_days = (alert_time - t0) / 86400.0
            if dt_days < 0 or dt_days > 14.0:
                continue

            # Evaluate GW 2D/3D posteriors at transient position
            cred_level, dp_dv = gw.evaluate_candidate(
                ra_deg, dec_deg, host_dl if host_dl is not None else 100.0
            ) #[cite: 4]

            # Spatial Gate: Must lie within the 90% credible sky contour
            if cred_level <= 0.9: #[cite: 2, 4]
                if dp_dv > highest_dp_dv:
                    highest_dp_dv = dp_dv
                    best_match = {
                        "event_id": event_id,
                        "cred_level": cred_level,
                        "dp_dv": dp_dv,
                    }

        if best_match is None:
            return self.get_fallback_features()

        # Query local host galaxy environment in GLADE+
        host_info = self.find_top_host_galaxy(
            ra_deg, dec_deg, self.active_gw_events[best_match["event_id"]]["gw_obj"]
        )

        return {
            "has_gw_coincidence": True,
            "gw_event_id": best_match["event_id"],
            "gw_cred_level": float(best_match["cred_level"]),
            "dp_dv": float(best_match["dp_dv"]),
            "best_host_dist_mpc": host_info["dist_mpc"],
            "best_host_K_mag": host_info["K_mag"],
            "S_gal": host_info["S_gal"],
        }

    def find_top_host_galaxy(self, ra_deg: float, dec_deg: float, gw: GW_data) -> dict:
        """Looks up nearest host candidate pixel and computes spatial-luminosity score S_gal.""" #[cite: 1, 2, 4]
        pix = ah.lonlat_to_healpix(
            ra_deg * u.deg, dec_deg * u.deg, nside=self.target_nside, order="nested"
        )

        if pix not in self.pixel_map: #[cite: 2]
            return {"dist_mpc": -1.0, "K_mag": 99.0, "S_gal": 0.0}

        candidate_indices = self.pixel_map[pix] #[cite: 2]
        
        # Extract galaxy positions and magnitudes
        xyz = self.glade_dict["tree"].data[candidate_indices] #[cite: 2, 3]
        sel_ra = (np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))) % 360 #[cite: 1, 2]
        sel_dec = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))) #[cite: 1, 2]
        sel_dl = self.glade_dict["d_L"][candidate_indices] #[cite: 2, 3]
        sel_K = self.glade_dict["K_mag"][candidate_indices] #[cite: 2, 3]

        # Evaluate 3D probability density per galaxy
        cred_levels, dp_dv = gw.evaluate_batch(sel_ra, sel_dec, sel_dl) #[cite: 2, 4]

        # Compute luminosity-weighted score S_gal
        dist_mod = 5 * np.log10(sel_dl * 1e6 / 10) #[cite: 2]
        L_K = 10 ** (-0.4 * (sel_K - dist_mod)) #[cite: 2]
        S_gal = dp_dv * L_K #[cite: 2]

        top_idx = np.argmax(S_gal)
        return {
            "dist_mpc": float(sel_dl[top_idx]),
            "K_mag": float(sel_K[top_idx]),
            "S_gal": float(S_gal[top_idx]),
        }