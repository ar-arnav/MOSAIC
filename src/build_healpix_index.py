import pickle
from pathlib import Path
import astropy.units as u
import astropy_healpix as ah
import numpy as np

glade_pkl = Path("~/MOSAIC/data/processed/GLADE+/glade_ckdtree.pkl").expanduser()
out_path = Path("~/MOSAIC/data/processed/GLADE+").expanduser()

NSIDE = 128


def load_glade_pkl(glade_pkl):
    with open(glade_pkl, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    glade_dict = load_glade_pkl(glade_pkl)
    print("Building HEALPix spatial index...")

    xyz = glade_dict["tree"].data

    # Reconstruct RA and Dec in DEGREES with domain safety
    ra_deg = (np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))) % 360
    dec_deg = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))

    # Compute HEALPix pixel IDs with explicit Astropy units
    galaxy_pixels = ah.lonlat_to_healpix(
        ra_deg * u.deg, dec_deg * u.deg, nside=NSIDE, order="nested"
    )

    # Fast vectorized grouping via unique inverse sorting
    unique_pixels, inverse = np.unique(galaxy_pixels, return_inverse=True)
    sort_idx = np.argsort(inverse)
    sorted_inv = inverse[sort_idx]

    # Find array indices where pixel ID transitions occur
    changes = np.where(np.diff(sorted_inv) > 0)[0] + 1
    bounds = np.concatenate([[0], changes, [len(sorted_inv)]])

    pixel_map = {}
    for i in range(len(bounds) - 1):
        pixel_map[unique_pixels[i]] = sort_idx[bounds[i] : bounds[i + 1]]

    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"glade_healpix_nside{NSIDE}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(pixel_map, f)

    print(f"Saved {len(pixel_map)} populated pixels to {out_file}")