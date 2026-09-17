from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import astropy_healpix as ah

from gw_processor import GW_data

glade_ckdtree_pkl = Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl").expanduser()
glade_healpix_pkl = Path("~/MOSAIC/data/processed/GLADE+/glade_healpix_nside128.pkl").expanduser()
TARGET_NSIDE = 128

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_gw_active_pixels(gw, target_nside=128, max_cred=0.9):
    """
    Extracts nested HEALPix pixel IDs at target_nside matching 
    the GW sky map's multi-resolution UNIQ 90% credible contour.
    """
    mask = gw.credible_levels <= max_cred
    active_uniq = gw.uniq[mask]
    
    active_pixels = set()
    for u in active_uniq:
        level, ipix = ah.uniq_to_level_ipix(u)
        nside = ah.level_to_nside(level)
        
        if nside == target_nside:
            active_pixels.add(ipix)
        elif nside < target_nside:
            # Subdivide parent pixel into child pixels at target_nside
            factor = (target_nside // nside) ** 2
            start = ipix * factor
            active_pixels.update(range(start, start + factor))
        else:
            # Degrade child pixel to parent pixel at target_nside
            factor = (nside // target_nside) ** 2
            active_pixels.add(ipix // factor)
            
    return active_pixels

def rank_galaxies(gw, glade_dict, pixel_map, top_n=100):
    print("Ranking galaxies...")
    
    # 1. Identify spatial pixels inside 90% GW contour
    active_pixels = get_gw_active_pixels(gw, target_nside=TARGET_NSIDE, max_cred=0.9)
    
    # 2. Extract catalog array indices from spatial hash map
    candidate_lists = [pixel_map[p] for p in active_pixels if p in pixel_map]
    if not candidate_lists:
        print("No galaxies found within the 90% credible region.")
        return pd.DataFrame()
        
    candidate_indices = np.unique(np.concatenate(candidate_lists))
    print(f"Pre-filtered {len(candidate_indices)} candidate galaxies from HEALPix hash map.")

    # 3. Reconstruct coordinates and distances ONLY for candidate subset
    xyz = glade_dict['tree'].data[candidate_indices]
    sel_ra = (np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))) % 360
    sel_dec = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    sel_dl = glade_dict['d_L'][candidate_indices]
    sel_K = glade_dict['K_mag'][candidate_indices]

    # 4. Evaluate GW 3D probabilities (vectorized over candidates)
    cred_levels, dp_dv = gw.evaluate_batch(sel_ra, sel_dec, sel_dl)

    # 5. Strict 90% credible level filter
    mask = cred_levels <= 0.9
    if not np.any(mask):
        return pd.DataFrame()

    sel_ra = sel_ra[mask]
    sel_dec = sel_dec[mask]
    sel_dl = sel_dl[mask]
    sel_cred = cred_levels[mask]
    sel_dp_dv = dp_dv[mask]
    sel_K = sel_K[mask]

    # 6. Distance Modulus & Luminosity Scoring
    dist_mod = 5 * np.log10(sel_dl * 1e6 / 10)
    L_K = 10**(-0.4 * (sel_K - dist_mod))
    S_gal = sel_dp_dv * L_K

    # 7. Build and sort output DataFrame
    glade_ranked = pd.DataFrame({
        'ra': sel_ra,
        'dec': sel_dec,
        'dist_mpc': sel_dl,
        'cred_level': sel_cred,
        'dp_dv': sel_dp_dv,
        'K_mag': sel_K,
        'L_K': L_K,
        'S_gal': S_gal
    }).sort_values('S_gal', ascending=False).head(top_n).reset_index(drop=True)

    return glade_ranked

if __name__ == "__main__":
    glade_dict = load_pkl(glade_ckdtree_pkl)
    pixel_map = load_pkl(glade_healpix_pkl)
    
    gw = GW_data(Path("~/MOSAIC/data/raw/GW/1.fits").expanduser())
    ranked = rank_galaxies(gw, glade_dict, pixel_map)
    print(ranked.head(10))