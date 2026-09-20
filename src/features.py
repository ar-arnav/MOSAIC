# src/features.py
import numpy as np

def format_spatial_features(spatial_dict: dict, eps: float = 1e-30) -> np.ndarray:
    """
    Transforms the raw spatial pipeline output dictionary into a static, 
    normalized float32 vector for the Stream 3 Fusion Network.
    
    Input: Dictionary from SpatialPipeline (contains keys like 'has_gw_coincidence', 'dp_dv', etc.)
    Output: Numpy array of shape (6,)
    """
    # 1. Convert Boolean coincidence flag to float (1.0 or 0.0)
    is_coinc = 1.0 if spatial_dict["has_gw_coincidence"] else 0.0
    
    # 2. Extract Credibility Level (from GW skymap)
    cred = float(spatial_dict["gw_cred_level"])
    
    # 3. Extract and Log-transform dp_dv (Probability Density)
    # We add 'eps' (a tiny number) to avoid log(0) errors
    dp_dv_raw = max(float(spatial_dict["dp_dv"]), 0.0)
    log_dp_dv = np.log10(dp_dv_raw + eps)
    
    # 4. Extract and Log-transform S_gal (Galaxy Score/Luminosity Density)
    # This acts as the 'log_sgal' feature
    sgal_raw = max(float(spatial_dict["S_gal"]), 0.0)
    log_sgal = np.log10(sgal_raw + eps)
    
    # 5. Extract Distance and Magnitude
    dist = float(spatial_dict["best_host_dist_mpc"])
    kmag = float(spatial_dict["best_host_K_mag"])

    # Pack into a 6-element array
    return np.array([is_coinc, cred, log_dp_dv, log_sgal, dist, kmag], dtype=np.float32)