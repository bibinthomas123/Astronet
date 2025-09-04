# Run this in your environment (Jupyter / Colab). Internet required.
# Install packages if needed:
# !pip install astroquery astropy pandas numpy

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\bibin.a.thomas\bazel_projects\final_planets.csv");

targets = df.to_dict('records')  # each row becomes a dict

# radius in arcsec to search for neighbors
search_radius_arcsec = 120.0   # 2 arcminutes
radius_deg = search_radius_arcsec / 3600.0

# output rows
rows = []

for t in targets:
    coord = SkyCoord(t['ra'], t['dec'], unit=(u.deg, u.deg), frame='icrs')
    # ADQL cone search using astroquery (Gaia DR3)
    query = f"""
SELECT TOP 50
    source_id,
    ra,
    dec,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    DISTANCE(
        POINT('ICRS', {t['ra']}, {t['dec']}),
        POINT('ICRS', ra, dec)
    ) * 3600 AS ang_dist_arcsec
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {t['ra']}, {t['dec']}, {radius_deg})
)
ORDER BY ang_dist_arcsec

        """
    job = Gaia.launch_job(query)
    results = job.get_results()
    # results is an astropy Table
    if len(results) == 0:
        nearest = None
    else:
        # nearest neighbor might be the target itself (very often source at tiny separation)
        # compute angular separations and pick the nearest *other* source (sep > 0.1")
        ras = results['ra'].data
        decs = results['dec'].data
        coords = SkyCoord(ras, decs, unit='deg')
        sep = coords.separation(coord).arcsec
        # convert to numpy arrays
        sep = np.array(sep)
        # find the first neighbor with sep > 0.1 arcsec (exclude exact match)
        idxs = np.where((sep > 0.5) & (results['phot_g_mean_mag'] < 18))[0]
        if len(idxs)==0:
            nearest = None
        else:
            i = idxs[1]
            row = results[i]
            nearest = {
                'source_id': int(row['SOURCE_ID']),
                'ra': float(row['ra']),
                'dec': float(row['dec']),
                'G': float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] is not None else np.nan,
                'BP': float(row['phot_bp_mean_mag']) if row['phot_bp_mean_mag'] is not None else np.nan,
                'RP': float(row['phot_rp_mean_mag']) if row['phot_rp_mean_mag'] is not None else np.nan,
                'sep_arcsec': sep[i]
            }
    # compute expected shift if neighbor exists
    if nearest is None:
        expected_shift_mas = np.nan
        flux_ratio = np.nan
        neighbor_sep = np.nan
        neighbor_G = np.nan
    else:
        neighbor_sep = nearest['sep_arcsec']   # arcsec
        neighbor_G = nearest['G']
        # assume flux ratio f = 10^{-0.4*(G_neighbor - G_target)}; but we do not have target G in Gaia for some,
        # approximate target G by nearest Gaia source at sep ~ 0 (first row)
        if len(results)>0:
            # take first row as target
            targetG = float(results[0]['phot_g_mean_mag']) if results[0]['phot_g_mean_mag'] is not None else np.nan
        else:
            targetG = np.nan
        if np.isfinite(targetG) and np.isfinite(nearest['G']):
            f = 10**(-0.4*(nearest['G'] - targetG))
        else:
            f = 0.0
        
        flux_ratio = f
        depth_ppm = t['koi_depth']
        delta = depth_ppm * 1e-6
        flux_ratio = f
        # expected centroid shift in mas: sep (arcsec) * 1000 (mas/arcsec) * delta/(1+f)
        expected_shift_mas = neighbor_sep * 1000.0 * (delta/(1.0+f))
    rows.append({
        'koi': t['kepoi_name'],
        'kic': t['kic_id'],
        'ra': t['ra'],
        'dec': t['dec'],
        'nearest_sep_arcsec': neighbor_sep / 60,
        'nearest_G': neighbor_G,
        'flux_ratio': flux_ratio,
        'transit_depth_ppm': t['depth_ppm'],
        'expected_shift_mas': expected_shift_mas
    })

# produce DataFrame and write CSV
df_out = pd.DataFrame(rows)
# df_out.to_csv("centroid_neighbor_results.csv", index=False)
print(df_out.to_string(index=False))
