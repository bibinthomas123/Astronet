import os
from lightkurve import search_lightcurve

kepler_id = 11442793
prefix = str(kepler_id).zfill(9)[:4]
base_dir = "C:/Users/bibin.a.thomas/bazel_projects/kepler"
target_dir = os.path.join(base_dir, prefix, str(kepler_id).zfill(9))
os.makedirs(target_dir, exist_ok=True)

# Download all long cadence light curves
search_result = search_lightcurve(f"KIC {kepler_id}", cadence="long")
print(f"Found {len(search_result)} light curves. Downloading...")

for lc in search_result:
    lc.download(download_dir=target_dir)

print(f"\n✅ All FITS segments saved to:\n{target_dir}")
