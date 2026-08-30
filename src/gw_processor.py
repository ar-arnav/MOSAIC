import os
from pathlib import Path
import numpy as np
import pandas as pd
import healpy as hp
from astropy.table import Table
import astropy_healpix as ah
from astropy.coordinates import SkyCoord


class GW_data:
    def __init__(self, GW_data_path, t0=None):
        self.GW_data = Table.read(GW_data_path)
        self.t0 = t0  # merger time, reserved for temporal gating

        uniq_raw = np.array(self.GW_data['UNIQ'], dtype=np.int64)
        sort_idx = np.argsort(uniq_raw)
        self.GW_data = self.GW_data[sort_idx]
        self.uniq = uniq_raw[sort_idx]

        level, ipix = ah.uniq_to_level_ipix(self.uniq)
        self.min_level = np.min(level)
        self.max_level = np.max(level)

        level_nside = ah.level_to_nside(level)
        level_pixels = ah.nside_to_pixel_area(level_nside)
        probdensity = self.GW_data['PROBDENSITY']
        self.pixel_prob = probdensity * level_pixels

        self.distmu = np.array(self.GW_data['DISTMU'], dtype=np.float64)
        self.distsig = (np.array(self.GW_data['DISTSIGMA'], dtype=np.float64)
                         if 'DISTSIGMA' in self.GW_data.colnames
                         else np.array(self.GW_data['DISTSIG'], dtype=np.float64))
        self.distnorm = np.array(self.GW_data['DISTNORM'], dtype=np.float64)
        self.has_valid_distance = (self.distmu > 0) & (self.distsig > 0) & (self.distnorm > 0)

        prob_sorted = np.argsort(self.pixel_prob)[::-1]
        cumsum_result = np.cumsum(self.pixel_prob[prob_sorted])
        total_prob = cumsum_result / cumsum_result[-1]
        self.credible_levels = np.zeros(len(self.pixel_prob))
        self.credible_levels[prob_sorted] = total_prob

        self._probdensity_vals = np.asarray(self.GW_data['PROBDENSITY'], dtype=np.float64)

    def evaluate_candidate(self, ra_deg, dec_deg, dist_mpc):
        row_idx = None

        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')

        nside = ah.level_to_nside(self.max_level)
        target_ipix = ah.lonlat_to_healpix(coord.ra, coord.dec, nside, order='nested')

        target_uniq = np.int64(ah.level_ipix_to_uniq(self.max_level, target_ipix))

        idx = np.searchsorted(self.uniq, target_uniq)

        if idx < len(self.uniq) and self.uniq[idx] == target_uniq:
            row_idx = idx
        else:
            curr_level = self.max_level
            curr_ipix = target_ipix

            while curr_level > self.min_level:
                curr_ipix = curr_ipix // 4
                curr_level -= 1

                curr_uniq = np.int64(ah.level_ipix_to_uniq(curr_level, curr_ipix))

                curr_idx = np.searchsorted(self.uniq, curr_uniq)
                if curr_idx < len(self.uniq) and self.uniq[curr_idx] == curr_uniq:
                    row_idx = curr_idx
                    break

        if row_idx is None:
            return 1.0, 0.0

        mu = self.distmu[row_idx]
        sigma = self.distsig[row_idx]
        norm = self.distnorm[row_idx]
        prob_2d = float(self._probdensity_vals[row_idx])

        if mu <= 0 or sigma <= 0 or norm <= 0:
            return self.credible_levels[row_idx], 0.0

        gaussian = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((dist_mpc - mu) / sigma) ** 2)
        dp_dv = norm * gaussian * prob_2d
        return self.credible_levels[row_idx], dp_dv

    def evaluate_batch(self, ra_deg, dec_deg, dist_mpc):
        """Vectorized version of evaluate_candidate. All inputs are arrays."""
        ra_deg = np.asarray(ra_deg)
        dec_deg = np.asarray(dec_deg)
        dist_mpc = np.asarray(dist_mpc)
        n = len(ra_deg)

        coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')
        nside = ah.level_to_nside(self.max_level)
        target_ipix = ah.lonlat_to_healpix(coords.ra, coords.dec, nside, order='nested')
        target_uniq = np.asarray(ah.level_ipix_to_uniq(self.max_level, target_ipix), dtype=np.int64)

        idx = np.searchsorted(self.uniq, target_uniq)
        idx = np.clip(idx, 0, len(self.uniq) - 1)
        found = self.uniq[idx] == target_uniq
        row_idx = np.where(found, idx, -1)

        # ============================================================
        # FIX: Filter BEFORE decrementing to avoid invalid levels
        # ============================================================
        unresolved = ~found
        curr_ipix = target_ipix.copy()
        curr_level = np.full(n, self.max_level, dtype=np.int64)

        while unresolved.any() and curr_level.max() >= self.min_level:
            # Filter out entries that have reached minimum level
            # BEFORE decrementing, to avoid passing invalid (negative) levels
            unresolved &= (curr_level > self.min_level)
            if not unresolved.any():
                break

            curr_ipix[unresolved] //= 4
            curr_level[unresolved] -= 1
            # Now curr_level[unresolved] is guaranteed >= min_level

            curr_uniq = np.asarray(
                ah.level_ipix_to_uniq(curr_level[unresolved], curr_ipix[unresolved]),
                dtype=np.int64
            )
            cidx = np.searchsorted(self.uniq, curr_uniq)
            cidx = np.clip(cidx, 0, len(self.uniq) - 1)
            hit = self.uniq[cidx] == curr_uniq
            hit_global = np.where(unresolved)[0][hit]
            row_idx[hit_global] = cidx[hit]
            unresolved[hit_global] = False

        cred_level = np.where(row_idx >= 0, self.credible_levels[np.clip(row_idx, 0, None)], 1.0)
        dp_dv = np.zeros(n)

        valid = row_idx >= 0
        vi = row_idx[valid]
        mu, sigma, norm = self.distmu[vi], self.distsig[vi], self.distnorm[vi]
        prob_2d = self._probdensity_vals[vi]
        has_dist = (mu > 0) & (sigma > 0) & (norm > 0)

        d = dist_mpc[valid]
        gaussian = np.zeros(len(vi))
        gaussian[has_dist] = (1 / (np.sqrt(2 * np.pi) * sigma[has_dist])
                               * np.exp(-0.5 * ((d[has_dist] - mu[has_dist]) / sigma[has_dist]) ** 2))
        dp_dv_valid = np.zeros(len(vi))
        dp_dv_valid[has_dist] = norm[has_dist] * gaussian[has_dist] * prob_2d[has_dist]

        dp_dv[valid] = dp_dv_valid
        return cred_level, dp_dv

    def filter_candidates(self, ra_deg, dec_deg, dist_mpc, dp_dv_threshold=None, credible_level_threshold=0.9):

        ra_deg = np.asarray(ra_deg)
        dec_deg = np.asarray(dec_deg)
        dist_mpc = np.asarray(dist_mpc)

        cred_levels, dp_dv = self.evaluate_batch(ra_deg, dec_deg, dist_mpc)

        results = pd.DataFrame({
            'ra': ra_deg,
            'dec': dec_deg,
            'dist_mpc': dist_mpc,
            'cred_level': cred_levels,
            'dp_dv': dp_dv
        })

        mask = cred_levels <= credible_level_threshold

        if dp_dv_threshold is not None:
            mask = mask & (dp_dv >= dp_dv_threshold)

        filtered = results[mask].sort_values('dp_dv', ascending=False).reset_index(drop=True)

        return filtered
