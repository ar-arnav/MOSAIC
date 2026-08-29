import os
from pathlib import Path
import numpy as np
import pandas as pd
import healpy as hp
from astropy.table import Table
import astropy_healpix as ah
from astropy.coordinates import SkyCoord


class GW_data:
    def __init__(self, GW_data_path):
        self.GW_data = Table.read(GW_data_path)

        self.uniq = np.array(self.GW_data['UNIQ'], dtype=np.int64)
        level, ipix = ah.uniq_to_level_ipix(self.uniq)
        self.min_level = np.min(level)
        self.max_level = np.max(level)

        unique_levels, counts = np.unique(level, return_counts=True)

        level_nside = ah.level_to_nside(level)
        level_pixels = ah.nside_to_pixel_area(level_nside)

        probdensity = self.GW_data['PROBDENSITY'] 

        self.pixel_prob = probdensity * level_pixels

        # --- EXPOSE 3D DISTANCE POSTERIORS ---
        self.distmu = np.array(self.GW_data['DISTMU'], dtype=np.float64)
        
        if 'DISTSIGMA' in self.GW_data.colnames:
            self.distsig = np.array(self.GW_data['DISTSIGMA'], dtype=np.float64)
        else:
            self.distsig = np.array(self.GW_data['DISTSIG'], dtype=np.float64)
        
        self.distnorm = np.array(self.GW_data['DISTNORM'], dtype=np.float64)
        
        # Pre-compute mask for pixels with VALID distance information
        self.has_valid_distance = (
            (self.distmu > 0) & 
            (self.distsig > 0) & 
            (self.distnorm > 0)
        )
        # ------------------------------------

        prob_sorted = np.argsort(self.pixel_prob)[::-1]
        cumsum_result = np.cumsum(self.pixel_prob[prob_sorted])
        total_prob = cumsum_result / cumsum_result[-1]

        self.credible_levels = np.zeros(len(self.pixel_prob))
        self.credible_levels[prob_sorted] = total_prob

    def evaluate_candidate(self, ra_deg, dec_deg, dist_mpc):
        row_idx = None
        
        Skycoord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')
    
        nside = ah.level_to_nside(self.max_level)
        target_ipix = ah.lonlat_to_healpix(Skycoord.ra, Skycoord.dec, nside, order='nested')
    
        target_uniq = ah.level_ipix_to_uniq(self.max_level, target_ipix)
    
        idx = np.searchsorted(self.uniq, target_uniq)
    
        if idx < len(self.uniq) and self.uniq[idx] == target_uniq:
            row_idx = idx
        else:
            curr_level = self.max_level
            curr_ipix = target_ipix
    
            while curr_level >= self.min_level:
                curr_ipix = curr_ipix // 4
                curr_level -= 1
    
                curr_uniq = ah.level_ipix_to_uniq(curr_level, curr_ipix)
                
                curr_idx = np.searchsorted(self.uniq, curr_uniq)
                if curr_idx < len(self.uniq) and self.uniq[curr_idx] == curr_uniq:
                    row_idx = curr_idx
                    break

        if row_idx is None:
            return 1.0, 0.0

        mu = self.distmu[row_idx]
        sigma = self.distsig[row_idx]
        norm = self.distnorm[row_idx]
        prob_2d = self.GW_data['PROBDENSITY'][row_idx]
        
        # Handle sentinel values (-1 means no distance info)
        if mu <= 0 or sigma <= 0 or norm <= 0:
            return self.credible_levels[row_idx], 0.0

        gaussian = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((dist_mpc - mu) / sigma) ** 2)
        dp_dv = norm * gaussian * prob_2d
        return self.credible_levels[row_idx], dp_dv