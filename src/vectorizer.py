import json
import numpy as np
import pandas as pd
import os

class RTFVectorizer:
    def __init__(self, stats_path="docs/Model W&B/RTF_MOSAIC/mosaic_rtf_stats.json"):

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found at {stats_path}. Run export_onnx.py first.")
            
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.max_len = 257
        self.n_bands = 3
        self.in_channels = 18

        self.base_keys = ["log_dt", "log_dt_prev", "logflux", "logflux_err"]
        self.alert_meta_keys = [
            "sharpnr", "scorr", "diffmaglim", "sky", "sigmapsf",
            "chinr", "rb", "chipsf", "distnr", "magnr", "fwhm"
        ]

    def _norm(self, x: np.ndarray, key: str) -> np.ndarray:
        
        stat = self.stats.get(key, {"median": 0.0, "iqr": 1.0})
        med = stat["median"]
        iqr = stat["iqr"]
        return np.clip((x - med) / iqr, -10, 10)

    def vectorize(self, df: pd.DataFrame):
        
        # 1. Sort by Julian Date and filter bands [1, 2, 3]
        df = df.sort_values("jd")
        df = df[df["fid"].isin([1, 2, 3])]
        
        if len(df) < 3:
            raise ValueError(f"Light curve too short (L={len(df)}). Minimum 3 points required.")
            
        if len(df) > self.max_len:
            df = df.iloc[:self.max_len]

        L = len(df)
        
        # 2. Extract raw arrays
        jds = df["jd"].values.astype(np.float64)
        mags = df["magpsf"].values.astype(np.float32)
        sigs = df["sigmapsf"].values.astype(np.float32)
        fids = df["fid"].values.astype(np.int64)

        # 3. Compute base time deltas and flux conversions
        dt = (jds - jds[0]).astype(np.float32)
        dtp = np.zeros(L, dtype=np.float32)
        dtp[1:] = np.diff(jds).astype(np.float32)

        # 4. Normalize base features (Channels 0-3)
        c0 = self._norm(np.log1p(dt), "log_dt")
        c1 = self._norm(np.log1p(dtp), "log_dt_prev")
        c2 = self._norm(-0.4 * mags, "logflux")
        c3 = self._norm(0.4 * sigs, "logflux_err")
        base = np.column_stack([c0, c1, c2, c3]).astype(np.float32)

        # 5. One-hot encode bands (Channels 4-6)
        one_hot = np.eye(self.n_bands, dtype=np.float32)[fids - 1]

            # 6. Extract and normalize metadata (Channels 7-17)
        meta = np.zeros((L, len(self.alert_meta_keys)), dtype=np.float32)
        for j, key in enumerate(self.alert_meta_keys):
            # Check if column exists, otherwise use zeros
            if key in df.columns:
                raw = df[key].fillna(0.0).values.astype(np.float32)
            else:
                raw = np.zeros(L, dtype=np.float32)
            
            # Safely normalize
            try:
                meta[:, j] = self._norm(raw, key)
            except:
                meta[:, j] = np.zeros(L, dtype=np.float32)

        # 7. Concatenate all features into the 18-channel matrix
        x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)

        # 8. Padding and Masking up to MAX_LEN (257)
        pad_mask = np.zeros(self.max_len, dtype=bool)
        if self.max_len - L > 0:
            padding = np.zeros((self.max_len - L, self.in_channels), dtype=np.float32)
            x = np.concatenate([x, padding], axis=0)
            pad_mask[L:] = True

        # 9. Add batch dimension -> Shape: (1, 257, 18) and (1, 257)
        return np.expand_dims(x, axis=0), np.expand_dims(pad_mask, axis=0)