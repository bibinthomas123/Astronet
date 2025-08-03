import os
import shutil
import requests
import io
import pandas as pd
import subprocess
from lightkurve import search_lightcurve
import time
import csv
from datetime import datetime

# === CONFIGURATION ===
BASE_DIR = "C:/Users/bibin.a.thomas/bazel_projects"
KEPLER_DIR = os.path.join(BASE_DIR, "kepler")
MODEL_DIR = os.path.join(BASE_DIR, "exoplanet-ml/model")

# === KIC IDs TO TEST ===
KEPLER_IDS = [
    11442793, 8480285, 11568987, 4852528, 5972334, 10337517, 11030475, 4548011,
    3832474, 11669239, 10130039, 5351250, 5301750, 8282651, 6786037, 6690082,
    9896018, 11450414, 6021275, 4851530, 8804283, 3323887, 6508221, 9006186,
    12061969, 11074178, 8397947, 11968463, 10600261, 3323887  # Note: 3323887 appears twice
]

# Remove duplicates while preserving order
UNIQUE_KEPLER_IDS = list(dict.fromkeys(KEPLER_IDS))

def download_light_curves(kepid):
    """Download light curves for a given Kepler ID"""
    print(f"📥 Downloading light curves for KIC {kepid}...")
    prefix = str(kepid).zfill(9)[:4]
    target_dir = os.path.join(KEPLER_DIR, prefix, str(kepid).zfill(9))
    os.makedirs(target_dir, exist_ok=True)

    try:
        search_result = search_lightcurve(f"KIC {kepid}", cadence="long")
        if len(search_result) == 0:
            print(f"❌ No light curve found for KIC {kepid}")
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
    
    except Exception as e:
        print(f"❌ Error downloading light curves for KIC {kepid}: {str(e)}")
        return None

def get_transit_params_from_archive(kepid):
    """Fetch TCE parameters from NASA Exoplanet Archive"""
    print(f"🔍 Fetching TCE parameters for KIC {kepid}...")
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_tce+where+kepid={kepid}&format=csv"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"❌ Failed to fetch data from NASA Exoplanet Archive for KIC {kepid}")
            return None
        
        df = pd.read_csv(io.StringIO(response.text))
        if df.empty:
            print(f"❌ No TCE found for KIC {kepid}")
            return None
        
        # Use first TCE if multiple exist
        row = df.iloc[0]
        
        params = {
            "period": round(row["tce_period"], 5),
            "t0": round(row["tce_time0bk"], 5),
            "duration": round(row["tce_duration"], 5)
        }
        
        print(f"✅ Found TCE for KIC {kepid}: Period={params['period']}, T0={params['t0']}, Duration={params['duration']}")
        return params
    
    except Exception as e:
        print(f"❌ Error fetching TCE parameters for KIC {kepid}: {str(e)}")
        return None

def run_prediction(kepid, period, t0, duration):
    """Run AstroNet prediction for given parameters"""
    print(f"\n🚀 Running AstroNet prediction for KIC {kepid}...")

    image_file = os.path.join(BASE_DIR, "exoplanet-ml", "predicted_images", f"kepler-{kepid}.png")
    prediction_file = os.path.join(BASE_DIR, 'exoplanet-ml', f'prediction_{kepid}.txt')

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
        f"--output_prediction_file={prediction_file}"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Prediction completed for KIC {kepid}")
            return prediction_file
        else:
            print(f"❌ Prediction failed for KIC {kepid}: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ Prediction timed out for KIC {kepid}")
        return None
    except Exception as e:
        print(f"❌ Error running prediction for KIC {kepid}: {str(e)}")
        return None

def parse_prediction_result(prediction_file):
    """Parse the prediction result from the output file"""
    if not os.path.exists(prediction_file):
        return None, None
    
    try:
        with open(prediction_file, 'r') as f:
            content = f.read()
            
        # Extract prediction score and classification
        score = None
        classification = None
        
        for line in content.split('\n'):
            line = line.strip()
            if '=' in line and any(char.isdigit() for char in line):
                if 'KIC' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1].strip())
                        except:
                            pass
            elif 'planet candidate' in line.lower():
                if 'not' in line.lower():
                    classification = "Not a planet candidate"
                else:
                    classification = "Is a planet candidate"
        
        return score, classification
        
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        return None, None

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    csv_file = os.path.join(BASE_DIR, "exoplanet-ml", filename)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'KIC_ID', 'Status', 'Period', 'T0', 'Duration', 
            'Prediction_Score', 'Classification', 'Error_Message', 'Timestamp'
        ])
        
        # Write data
        for result in results:
            writer.writerow([
                result.get('kepid', ''),
                result.get('status', ''),
                result.get('period', ''),
                result.get('t0', ''),
                result.get('duration', ''),
                result.get('prediction_score', ''),
                result.get('classification', ''),
                result.get('error', ''),
                result.get('timestamp', '')
            ])
    
    print(f"📊 Results saved to: {csv_file}")
    return csv_file

def test_kepler_ids_batch():
    """Test all Kepler IDs in the list"""
    
    print("🚀 Starting Batch Kepler ID Testing")
    print("=" * 60)
    print(f"📋 Total KIC IDs to test: {len(UNIQUE_KEPLER_IDS)}")
    print(f"🎯 KIC IDs: {UNIQUE_KEPLER_IDS}")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(os.path.join(BASE_DIR, "exoplanet-ml", "predicted_images"), exist_ok=True)
    
    results = []
    successful_predictions = 0
    failed_predictions = 0
    
    start_time = time.time()
    
    for i, kepid in enumerate(UNIQUE_KEPLER_IDS, 1):
        print(f"\n📍 Processing {i}/{len(UNIQUE_KEPLER_IDS)}: KIC {kepid}")
        print("-" * 50)
        
        result = {
            'kepid': kepid,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Unknown',
            'error': None,
            'period': None,
            't0': None,
            'duration': None,
            'prediction_score': None,
            'classification': None
        }
        
        try:
            # Step 1: Download light curve
            light_curve_dir = download_light_curves(kepid)
            if not light_curve_dir:
                result['status'] = 'Failed'
                result['error'] = 'Light curve download failed'
                results.append(result)
                failed_predictions += 1
                continue

            # Step 2: Fetch transit parameters
            params = get_transit_params_from_archive(kepid)
            if not params:
                result['status'] = 'Failed'
                result['error'] = 'TCE parameters not found'
                results.append(result)
                failed_predictions += 1
                continue
            
            result['period'] = params['period']
            result['t0'] = params['t0']
            result['duration'] = params['duration']

            # Step 3: Run prediction
            prediction_file = run_prediction(kepid, params['period'], params['t0'], params['duration'])
            
            if not prediction_file:
                result['status'] = 'Failed'
                result['error'] = 'Prediction execution failed'
                results.append(result)
                failed_predictions += 1
                continue
            
            # Step 4: Parse results
            score, classification = parse_prediction_result(prediction_file)
            
            result['prediction_score'] = score
            result['classification'] = classification
            result['status'] = 'Success'
            
            successful_predictions += 1
            
            print(f"✅ KIC {kepid} completed successfully")
            if score is not None:
                print(f"   📊 Score: {score:.6f}")
            if classification:
                print(f"   🎯 Classification: {classification}")
                
        except Exception as e:
            print(f"❌ Unexpected error processing KIC {kepid}: {str(e)}")
            result['status'] = 'Failed'
            result['error'] = f'Unexpected error: {str(e)}'
            failed_predictions += 1
        
        results.append(result)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(UNIQUE_KEPLER_IDS) - i) * avg_time
        print(f"⏱️  Progress: {i}/{len(UNIQUE_KEPLER_IDS)} | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    return results, successful_predictions, failed_predictions

def print_summary(results, successful_predictions, failed_predictions):
    """Print comprehensive summary of results"""
    
    print("\n" + "=" * 80)
    print("🎯 BATCH TESTING COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    success_rate = (successful_predictions / total_tests * 100) if total_tests > 0 else 0
    
    # Overall statistics
    print(f"📊 Overall Statistics:")
    print(f"   Total KIC IDs tested: {total_tests}")
    print(f"   Successful predictions: {successful_predictions}")
    print(f"   Failed predictions: {failed_predictions}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Successful predictions summary
    successful_results = [r for r in results if r['status'] == 'Success' and r['prediction_score'] is not None]
    
    if successful_results:
        scores = [r['prediction_score'] for r in successful_results]
        planet_candidates = [r for r in successful_results if r['classification'] and 'Is a planet' in r['classification']]
        non_planets = [r for r in successful_results if r['classification'] and 'Not a planet' in r['classification']]
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Planet candidates: {len(planet_candidates)}")
        print(f"   Non-planet candidates: {len(non_planets)}")
        print(f"   Average score: {sum(scores)/len(scores):.4f}")
        print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
        
        # Top scoring candidates
        print(f"\n🏆 Top 5 Planet Candidates (Highest Scores):")
        planet_sorted = sorted(planet_candidates, key=lambda x: x['prediction_score'], reverse=True)
        for i, result in enumerate(planet_sorted[:5], 1):
            print(f"   {i}. KIC {result['kepid']}: {result['prediction_score']:.6f}")
        
        # Lowest scoring non-planets
        print(f"\n🔍 Lowest Scoring Non-Planets:")
        non_planet_sorted = sorted(non_planets, key=lambda x: x['prediction_score'])
        for i, result in enumerate(non_planet_sorted[:3], 1):
            print(f"   {i}. KIC {result['kepid']}: {result['prediction_score']:.6f}")
    
    # Failure analysis
    if failed_predictions > 0:
        print(f"\n❌ Failure Analysis:")
        failure_reasons = {}
        for result in results:
            if result['status'] == 'Failed':
                error = result['error'] or 'Unknown error'
                failure_reasons[error] = failure_reasons.get(error, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"   {reason}: {count} cases")
    
    print("\n" + "=" * 80)

def main():
    """Main execution function"""
    
    print("🌟 Kepler ID Batch Testing Script")
    print("🔭 Testing Enhanced CNN + BiLSTM + Attention Model")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run batch testing
    results, successful, failed = test_kepler_ids_batch()
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"kepler_batch_results_{timestamp}.csv"
    save_results_to_csv(results, csv_filename)
    
    # Print summary
    print_summary(results, successful, failed)
    
    print(f"\n🎉 Batch testing completed!")
    print(f"📁 Detailed results saved to CSV file")

if __name__ == "__main__":
    main()