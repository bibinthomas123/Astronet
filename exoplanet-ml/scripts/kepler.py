import os
import shutil
import requests
import io
import pandas as pd
import subprocess
from lightkurve import search_lightcurve

# === CONFIGURATION ===
BASE_DIR = "C:/Users/bibin.a.thomas/bazel_projects"
KEPLER_DIR = os.path.join(BASE_DIR, "kepler")
MODEL_DIR = os.path.join(BASE_DIR, "exoplanet-ml/model")

def download_light_curves(kepid):
    print(f"📥 Downloading light curves for KIC {kepid}...")
    prefix = str(kepid).zfill(9)[:4]
    target_dir = os.path.join(KEPLER_DIR, prefix, str(kepid).zfill(9))
    os.makedirs(target_dir, exist_ok=True)

    search_result = search_lightcurve(f"KIC {kepid}", cadence="long")
    if len(search_result) == 0:
        print("❌ No light curve found.")
        return None

    for lc in search_result:
        lc.download(download_dir=target_dir)

    # Move any .fits from subfolders up to target_dir
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".fits"):
                full_path = os.path.join(root, file)
                if root != target_dir:
                    shutil.move(full_path, os.path.join(target_dir, file))

    print(f"✅ Light curves saved to: {target_dir}")
    return target_dir
def get_transit_params_from_archive(kepid):
    print(f"🔍 Fetching TCE parameters for KIC {kepid}...")
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_tce+where+kepid={kepid}&format=csv"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("❌ Failed to fetch data from NASA Exoplanet Archive.")
        return None
    
    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        print(f"❌ No TCE found for KIC {kepid}.")
        return None
    
    row = df.iloc[0]
    
    print(f"✅ Found TCE for KIC {kepid}: Period={row['tce_period']}, T0={row['tce_time0bk']}, Duration={row['tce_duration']}")

    return {
        "period": round(row["tce_period"], 5),
        "t0": round(row["tce_time0bk"], 5),
        "duration": round(row["tce_duration"], 5)
    }

def run_prediction(kepid, period, t0, duration):
    print(f"\n🚀 Running AstroNet prediction for KIC {kepid}...\n")

    image_file = os.path.join(BASE_DIR,"exoplanet-ml","predicted_images", f"kepler-{kepid}.png")

    print(f"Prediction for KIC : {kepid}")

    command = [
        "bazel", "run", "//astronet:predict", "--",
        "--model=AstroCNNModel",
        f"--config_json={os.path.join(MODEL_DIR, 'config.json')}",
        f"--model_dir={MODEL_DIR}",
        f"--kepler_data_dir={KEPLER_DIR}",
        f"--kepler_id={kepid}",
        f"--period={period}",
        f"--t0={t0}",
        f"--duration={duration}",
        f"--output_image_file={image_file}",
    ]

    subprocess.run(command)


# === MAIN ===
if __name__ == "__main__":
    # kepid = input("🔭 Enter Kepler ID: ").strip()
    kepid = "10797460"  # Example Kepler ID, replace with user input if needed
 
    if not kepid.isdigit():
        print("❌ Invalid Kepler ID.")
        exit(1)

    kepid = int(kepid)

    # Step 1: Download light curve
    if not download_light_curves(kepid):
        exit(1)

    # Step 2: Fetch transit parameters
    params = get_transit_params_from_archive(kepid)
    if not params:
        exit(1)

    # Step 3: Run AstroNet prediction
    run_prediction(kepid, params['period'], params['t0'], params['duration'])
