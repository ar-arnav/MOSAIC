from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from spatial import SpatialPipeline
from features import format_spatial_features
from model import MOSAICFusionMLP


class MOSAICPipeline:
    def __init__(
        self,
        onnx_stream1_path: Path,
        glade_tree_path: Path,
        glade_hpx_path: Path,
        fusion_weights_path: Path | None = None,
    ):
        print("Initializing MOSAIC Engine...")
        onnx_stream1_path = Path(onnx_stream1_path).expanduser()
        glade_tree_path = Path(glade_tree_path).expanduser()
        glade_hpx_path = Path(glade_hpx_path).expanduser()

        self.ort_session = ort.InferenceSession(
            str(onnx_stream1_path),
            providers=["CPUExecutionProvider"],
        )

        self.spatial_engine = SpatialPipeline(glade_tree_path, glade_hpx_path)

        self.fusion_model = MOSAICFusionMLP(rtf_dim=128, spatial_dim=6)
        if fusion_weights_path is not None:
            fp = Path(fusion_weights_path).expanduser()
            if fp.exists():
                state = torch.load(fp, map_location="cpu", weights_only=False)
                # allow raw state_dict or {"model": ...}
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                self.fusion_model.load_state_dict(state, strict=False)
                print(f"  fusion weights: {fp}")
            else:
                print(f"  warning: fusion weights missing ({fp}) — random head")
        self.fusion_model.eval()

    def process_alert(
        self,
        rtf_matrix: np.ndarray,
        pad_mask: np.ndarray | None,
        ra: float,
        dec: float,
        alert_time: float,
    ) -> float:
        """
        rtf_matrix: (1, T, 18) or (T, 18)
        pad_mask:   (1, T) or (T,) bool, True = pad
        Returns P(KN) in [0, 1]
        """
        x = np.asarray(rtf_matrix, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        inputs = {self.ort_session.get_inputs()[0].name: x}
        if len(self.ort_session.get_inputs()) > 1 and pad_mask is not None:
            m = np.asarray(pad_mask)
            if m.ndim == 1:
                m = m[None, ...]
            inputs[self.ort_session.get_inputs()[1].name] = m.astype(bool)

        z_rtf = self.ort_session.run(None, inputs)[0]  # (1, 128)

        spatial_dict = self.spatial_engine.process_alert(ra, dec, alert_time)
        x_spatial = format_spatial_features(spatial_dict)

        with torch.no_grad():
            p = self.fusion_model(
                torch.from_numpy(z_rtf).float(),
                torch.from_numpy(x_spatial).unsqueeze(0).float(),
            )
            return float(p.item())