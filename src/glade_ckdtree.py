# Imports
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.spatial import cKDTree
import astropy_healpix as ah

# Initialize paths and needed columns
glade_path = Path("~/MOSAIC/data/raw/GLADE+/GLADE+.txt").expanduser()
output_path = Path("~/MOSAIC/data/processed/GLADE+").expanduser()
usecols = [8, 9, 18, 32, 33, 34, 35, 38]

# Function to convert degress to radians and get a column-wise stack of cartesian coordinates
def radec_to_cartesian_unit(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])


# Generate the cKDTree and pickle it
if __name__ == "__main__":

    # Read the downloaded GLADE+ catalog
    glade = pd.read_csv(glade_path, sep='\s+', usecols=usecols, na_values='null', header=None)

    # Process the data
    glade.columns = ['RA', 'Dec', 'K_mag', 'd_L', 'd_L_err', 'dist_flag', 'stellar_mass', 'merger_rate']

    # Add constraints
    glade = glade[glade['dist_flag'].notna() & (glade['dist_flag'] != 0)]

    glade = glade[glade['d_L'] < 1000]

    glade = glade.dropna(subset=['RA', 'Dec', 'd_L', 'K_mag'])

    # Run the funtion to get cartesian coordinates

    xyz = radec_to_cartesian_unit(glade['RA'].values, glade['Dec'].values)

    tree = cKDTree(xyz)
    print(f"KD-tree built with {len(glade)} galaxies")

    # Data to pickle
    data = {
        'tree': tree,
        'd_L': glade['d_L'].values,
        'd_L_err': glade['d_L_err'].values,
        'K_mag': glade['K_mag'].values,
        'stellar_mass': glade['stellar_mass'].values,
        'merger_rate': glade['merger_rate'].values,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / 'glade_ckdtree.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out_file}")