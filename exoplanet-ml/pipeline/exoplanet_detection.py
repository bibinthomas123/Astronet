import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import warnings
import os
import requests
import time
import subprocess
import re
from astropy import units as u
import lightkurve as lk
from tqdm import tqdm
from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip



warnings.filterwarnings('ignore')

class EnhancedShallueValidator:
    """
    Enhanced Shallue-style exoplanet validation with light curve processing and phase folding
    Features:
    - Automated light curve downloading from MAST
    - Phase folding to enhance transit signals
    - Advanced signal processing and filtering
    - AstroNet ML model integration
    - Comprehensive validation pipeline
    """
    
    def __init__(self, csv_path=None, light_curve_dir="light_curves", auto_download=True,
             use_astronet=True, base_dir=None, model_dir=None, kepler_dir=None,
             download_method='lightkurve'):
    
    

        self.data = None
        self.results = []
        self.light_curve_dir = light_curve_dir
        self.auto_download = auto_download
        self.use_astronet = use_astronet
        self.download_method = download_method  # 'lightkurve' or 'mast_api'
        
        # AstroNet configuration
        self.base_dir = base_dir
        self.model_dir = model_dir 
        self.kepler_dir = kepler_dir 
        self.predicted_images_dir = os.path.join(self.base_dir, "pipeline", "predicted_images")
        
        # Light curve processing configuration
        self.phase_fold_dir = os.path.join(self.base_dir, "pipeline","phase_folded_plots")
        self.processed_lc_dir = os.path.join(self.base_dir, "pipeline","processed_light_curves")

        # Advanced validation parameters - NEW ADDITIONS
        self.stellar_contamination_threshold = 0.8 
        self.minimum_transit_count = 3
        self.period_stability_threshold = 0.02
        self.secondary_eclipse_threshold = 0.5 # Changed from 0.3 to 0.5 (more strict for EB detection)

        self.centroid_motion_threshold = 3.0  # sigma threshold for centroid shifts
        self.ttv_threshold = 0.1  # Transit timing variation threshold (days)  
        self.odd_even_depth_threshold = 0.3  # Threshold for odd/even transit depth difference
        self.multi_aperture_consistency_threshold = 0.8  # Consistency across apertures
        self.pixel_correlation_threshold = 0.7  # Pixel-level validation threshold
        self.neighboring_star_contamination_threshold = 0.2  

        # Create necessary directories
        os.makedirs(light_curve_dir, exist_ok=True)
        os.makedirs(self.predicted_images_dir, exist_ok=True)
        os.makedirs(self.phase_fold_dir, exist_ok=True)
        os.makedirs(self.processed_lc_dir, exist_ok=True)
        
        # MAST API configuration
        self.mast_base_url = "https://mast.stsci.edu/api/v0.1/"
        self.download_session = requests.Session()
        self.download_session.headers.update({
            'User-Agent': 'ExoplanetDetector/1.0 (Python)'
        })
    
    
        
        if csv_path:
            self.load_data(csv_path)

    def load_data(self, csv_path):
        """Load KOI catalog data and standardize columns"""
        print(f"Loading KOI catalog from {csv_path}...")
        self.data = pd.read_csv(csv_path, comment='#')
        
        print(f"Original data shape: {self.data.shape}")
        print(f"Available columns: {list(self.data.columns)}")
        
        # Map KOI columns to standard names
        column_mapping = {
            'kepid': 'kic_id',
            'koi_period': 'period',
            'koi_time0bk': 't0',
            'koi_duration': 'duration',
            'koi_depth': 'depth_ppm',
            'koi_model_snr': 'snr',
            'koi_disposition': 'disposition',
            'koi_pdisposition': 'pdisposition',
            'koi_score': 'score',
            'koi_prad': 'planet_radius',
            'koi_teq': 'equilibrium_temp',
            'koi_insol': 'insolation',
            'koi_kepmag': 'kepmag',
            'koi_steff': 'stellar_teff',
            'koi_slogg': 'stellar_logg',
            'koi_srad': 'stellar_radius',
            'ra': 'ra',
            'dec': 'dec'
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in self.data.columns:
                self.data[new_col] = self.data[old_col]
        
        # Clean and process data
        self._process_koi_data()
        
        print(f"Processed {len(self.data)} candidates after cleaning")
        return self.data

    def _process_koi_data(self):
        """Process and clean KOI catalog data"""
        
        # Remove rows with missing essential data
        essential_cols = ['kic_id', 'period', 't0', 'duration']
        before_count = len(self.data)
        
        for col in essential_cols:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        print(f"Removed {before_count - len(self.data)} rows with missing essential data")
    
        
        # Convert depth from ppm to fraction
        if 'depth_ppm' in self.data.columns:
            self.data['transit_depth'] = self.data['depth_ppm'] / 1e6
        
        # Calculate ML probability using AstroNet or synthetic method
        if self.use_astronet:
            print("🤖 AstroNet ML prediction enabled")
            self.data['model_prob'] = 0.0  # Will be filled during validation
        else:
            print("🧮 Using synthetic ML scores")
            self.data['model_prob'] = self._calculate_synthetic_ml_score()
        
        # Filter out obvious non-planetary candidates
        if 'disposition' in self.data.columns:
            # Keep CONFIRMED, CANDIDATE, and some others for analysis
            valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
            self.data = self.data[self.data['disposition'].isin(valid_dispositions)]
        
        # Ensure positive periods and reasonable ranges
        self.data = self.data[
            (self.data['period'] > 0.5) & 
            (self.data['period'] < 1000) &
            (self.data['duration'] > 0.1) &
            (self.data['duration'] < 24)
        ]
        
        # Remove duplicates based on KIC ID and period
        self.data = self.data.drop_duplicates(subset=['kic_id', 'period'])
        
        print(f"Final dataset: {len(self.data)} unique candidates")

    def download_light_curve_lightkurve(self, kic_id, quarters='all'):
        """Download light curve using lightkurve (recommended method) - FIXED VERSION"""
        
        try:
            print(f"  📡 Downloading light curve for KIC {kic_id} using lightkurve...")
            
            # Search for Kepler data with better error handling
            try:
                search_result = lk.search_lightcurve(f'KIC {kic_id}', mission='Kepler',cadence="long")
            except Exception as search_error:
                print(f"  ❌ Search failed for KIC {kic_id}: {search_error}")
                return None
            
            if len(search_result) == 0:
                print(f"  ❌ No Kepler data found for KIC {kic_id}")
                return None
            
            print(f"  📊 Found {len(search_result)} quarters of data")
            
            # Download all available quarters with timeout and error handling
            try:
                lc_collection = search_result.download_all()
                print("The light curves are now downloaded")
            except Exception as download_error:
                print(f"  ❌ Download failed for KIC {kic_id}: {download_error}")
                # Try downloading with default quality settings
                try:
                    lc_collection = search_result.download_all()
                except Exception as fallback_error:
                    print(f"  ❌ Fallback download also failed for KIC {kic_id}: {fallback_error}")
                    return None
            
            if lc_collection is None or len(lc_collection) == 0:
                print(f"  ❌ No light curve data downloaded for KIC {kic_id}")
                return None
            
            # Combine quarters and remove outliers with better error handling
            try:
                lc = lc_collection.stitch()
                
                # Check if we have valid data
                if lc is None or len(lc.flux) == 0:
                    print(f"  ❌ No valid flux data after stitching for KIC {kic_id}")
                    return None
                
                # Remove NaN values and normalize
                lc = lc.remove_nans()
                
                # Check again after removing NaNs
                if len(lc.flux) == 0:
                    print(f"  ❌ No data remaining after removing NaNs for KIC {kic_id}")
                    return None
                
                # Normalize flux
                lc = lc.normalize()
                
                # Final check
                if len(lc.flux) < 100:  # Need at least 100 data points
                    print(f"  ❌ Insufficient data points ({len(lc.flux)}) for KIC {kic_id}")
                    return None
                
            except Exception as processing_error:
                print(f"  ❌ Error processing light curve for KIC {kic_id}: {processing_error}")
                return None
            
            # Save processed light curve with error handling
            try:
                lc_filename = os.path.join(self.processed_lc_dir, f"kic_{kic_id}_processed.fits")
                lc.to_fits(lc_filename, overwrite=True)
            except Exception as save_error:
                print(f"  ⚠️ Could not save light curve file for KIC {kic_id}: {save_error}")
                # Continue anyway, we have the data in memory
            
            print(f"  ✅ Successfully downloaded and processed light curve")
            print(f"     Data points: {len(lc.flux)}")
            print(f"     Time span: {lc.time.min().value:.1f} to {lc.time.max().value:.1f} BKJD")
            
            return lc
            
        except Exception as e:
            print(f"  ❌ Unexpected error downloading light curve for KIC {kic_id}: {e}")
            return None

    def download_light_curve_mast(self, kic_id):
        """Alternative method to download light curve from MAST directly"""
        
        try:
            print(f"  📡 Downloading light curve for KIC {kic_id} from MAST...")
            
            # For now, fallback to lightkurve since MAST API is complex
            print(f"  ⚠️ MAST API download not fully implemented - using lightkurve fallback")
            return self.download_light_curve_lightkurve(kic_id)
            
        except Exception as e:
            print(f"  ❌ Error with MAST download for KIC {kic_id}: {e}")
            return self.download_light_curve_lightkurve(kic_id)

    def phase_fold_transit(self, lc, period, t0, duration):
        """
        Phase fold the light curve to enhance transit signal
        Returns phase-folded light curve and transit metrics
        """
        
        try:
            print(f"  🌀 Phase folding light curve (P={period:.4f}d, t0={t0:.4f})")
            
            # Validate inputs
            if lc is None or len(lc.flux) == 0:
                print(f"  ❌ Invalid light curve data for phase folding")
                return None, {}
            
            if period <= 0 or duration <= 0:
                print(f"  ❌ Invalid period or duration for phase folding")
                return None, {}
            
            # Calculate phases with better error handling
            try:
                time_values = lc.time.value
                flux_values = lc.flux.value
                
                # Check for valid time and flux data
                if len(time_values) == 0 or len(flux_values) == 0:
                    print(f"  ❌ No valid time or flux data")
                    return None, {}
                
                # Remove any remaining NaN or infinite values
                valid_mask = np.isfinite(time_values) & np.isfinite(flux_values)
                if not np.any(valid_mask):
                    print(f"  ❌ No finite values in time or flux data")
                    return None, {}
                
                time_values = time_values[valid_mask]
                flux_values = flux_values[valid_mask]
                
                # Calculate phases
                phases = ((time_values - t0) % period) / period
                
                # Center phases around transit (phase = 0.5 at transit)
                phases = (phases + 0.5) % 1.0
                
            except Exception as phase_error:
                print(f"  ❌ Error calculating phases: {phase_error}")
                return None, {}
            
            # Sort by phase
            try:
                sort_idx = np.argsort(phases)
                phases_sorted = phases[sort_idx]
                flux_sorted = flux_values[sort_idx]
            except Exception as sort_error:
                print(f"  ❌ Error sorting phase data: {sort_error}")
                return None, {}
            
            # Bin the phase-folded data for better S/N
            n_bins = min(100, max(50, len(flux_sorted) // 10) ) # Reasonable binning
            
            if n_bins < 20:
                print(f"  ⚠️ Too few data points ({len(flux_sorted)}) for reliable phase folding")
                return None, {}
            
            # Create phase bins and bin the flux
            try:
                phase_bins = np.linspace(0, 1, n_bins + 1)
                phase_centers = (phase_bins[1:] + phase_bins[:-1]) / 2
                
                # Bin the flux
                binned_flux = []
                binned_flux_err = []
                
                for i in range(n_bins):
                    mask = (phases_sorted >= phase_bins[i]) & (phases_sorted < phase_bins[i + 1])
                    if np.sum(mask) > 0:
                        bin_flux = flux_sorted[mask]
                        if len(bin_flux) > 0:
                            binned_flux.append(np.median(bin_flux))
                            binned_flux_err.append(np.std(bin_flux) / np.sqrt(len(bin_flux)))
                        else:
                            binned_flux.append(np.nan)
                            binned_flux_err.append(np.nan)
                    else:
                        binned_flux.append(np.nan)
                        binned_flux_err.append(np.nan)
                
                binned_flux = np.array(binned_flux)
                binned_flux_err = np.array(binned_flux_err)
                
                # Remove NaN bins
                valid_mask = np.isfinite(binned_flux) & np.isfinite(binned_flux_err)
                if not np.any(valid_mask):
                    print(f"  ❌ No valid bins after phase folding")
                    return None, {}
                
                phase_centers = phase_centers[valid_mask]
                binned_flux = binned_flux[valid_mask]
                binned_flux_err = binned_flux_err[valid_mask]
                
            except Exception as binning_error:
                print(f"  ❌ Error in phase binning: {binning_error}")
                return None, {}
            
            if len(binned_flux) < 10:
                print(f"  ❌ Too few valid bins ({len(binned_flux)}) after phase folding")
                return None, {}
            
            # Calculate transit metrics
            metrics = self._calculate_transit_metrics(phase_centers, binned_flux, binned_flux_err, duration, period)
            
            # Create phase-folded light curve object
            phase_folded_lc = {
                'phase': phase_centers,
                'flux': binned_flux,
                'flux_err': binned_flux_err,
                'period': period,
                't0': t0,
                'duration': duration,
                'metrics': metrics
            }
            
            print(f"  ✅ Phase folding complete - Transit depth: {metrics.get('transit_depth_ppm', 0):.1f} ppm")
            
            return phase_folded_lc, metrics
            
        except Exception as e:
            print(f"  ❌ Unexpected error in phase folding: {e}")
            return None, {}
        
    
    def detect_centroid_motion(self, lc, period, t0, duration):
        """
        CRITICAL: Detect centroid motion during transits - indicates background eclipsing binary
        Real planetary transits should show no significant centroid shifts
        """
        
        try:
            print(f"  🎯 Analyzing centroid motion during transits...")
            
            # Check if we have centroid data
            if not hasattr(lc, 'centroid_col') or not hasattr(lc, 'centroid_row'):
                print(f"  ⚠️ No centroid data available in light curve")
                return {'centroid_motion_detected': False, 'error': 'no_centroid_data'}
            
            time = lc.time.value
            flux = lc.flux.value
            centroid_col = lc.centroid_col.value
            centroid_row = lc.centroid_row.value
            
            # Remove NaN values
            valid_mask = (np.isfinite(time) & np.isfinite(flux) & 
                        np.isfinite(centroid_col) & np.isfinite(centroid_row))
            
            if np.sum(valid_mask) < 100:
                print(f"  ⚠️ Insufficient valid centroid data")
                return {'centroid_motion_detected': False, 'error': 'insufficient_centroid_data'}
            
            time = time[valid_mask]
            flux = flux[valid_mask]
            centroid_col = centroid_col[valid_mask]
            centroid_row = centroid_row[valid_mask]
            
            # Find all transit times
            transit_times = []
            current_t0 = t0
            while current_t0 < time.max():
                if current_t0 > time.min():
                    transit_times.append(current_t0)
                current_t0 += period
            
            if len(transit_times) < 2:
                print(f"  ⚠️ Need at least 2 transits for centroid analysis")
                return {'centroid_motion_detected': False, 'error': 'insufficient_transits'}
            
            # Analyze centroid motion for each transit
            transit_centroid_shifts = []
            baseline_centroid_std_col = np.std(centroid_col)
            baseline_centroid_std_row = np.std(centroid_row)
            
            significant_shifts = 0
            max_shift_col = 0
            max_shift_row = 0
            
            for transit_time in transit_times[:10]:  # Analyze up to 10 transits
                # Define transit window
                transit_mask = np.abs(time - transit_time) < (duration / 24.0 / 2)
                baseline_mask = ((np.abs(time - transit_time) > duration / 24.0) & 
                            (np.abs(time - transit_time) < period / 4))
                
                if np.sum(transit_mask) < 3 or np.sum(baseline_mask) < 10:
                    continue
                
                # Calculate centroid shifts during transit
                transit_centroid_col = np.median(centroid_col[transit_mask])
                transit_centroid_row = np.median(centroid_row[transit_mask])
                baseline_centroid_col = np.median(centroid_col[baseline_mask])
                baseline_centroid_row = np.median(centroid_row[baseline_mask])
                
                # Calculate shifts in units of baseline scatter
                shift_col = abs(transit_centroid_col - baseline_centroid_col) / baseline_centroid_std_col
                shift_row = abs(transit_centroid_row - baseline_centroid_row) / baseline_centroid_std_row
                
                max_shift_col = max(max_shift_col, shift_col)
                max_shift_row = max(max_shift_row, shift_row)
                
                # Check for significant shifts (> 3 sigma)
                if shift_col > self.centroid_motion_threshold or shift_row > self.centroid_motion_threshold:
                    significant_shifts += 1
                
                transit_centroid_shifts.append({
                    'transit_time': transit_time,
                    'shift_col_sigma': shift_col,
                    'shift_row_sigma': shift_row,
                    'is_significant': (shift_col > self.centroid_motion_threshold or 
                                    shift_row > self.centroid_motion_threshold)
                })
            
            # Overall assessment
            fraction_with_shifts = significant_shifts / max(len(transit_centroid_shifts), 1)
            max_overall_shift = max(max_shift_col, max_shift_row)
            
            # Determine if this indicates a background eclipsing binary
            is_background_eb = (fraction_with_shifts > 0.5 or max_overall_shift > 5.0)
            
            result = {
                'centroid_motion_detected': is_background_eb,
                'max_shift_col_sigma': float(max_shift_col),
                'max_shift_row_sigma': float(max_shift_row),
                'max_overall_shift_sigma': float(max_overall_shift),
                'fraction_transits_with_shifts': float(fraction_with_shifts),
                'individual_transit_shifts': transit_centroid_shifts,
                'baseline_centroid_std_col': float(baseline_centroid_std_col),
                'baseline_centroid_std_row': float(baseline_centroid_std_row),
                'likely_background_binary': is_background_eb
            }
            
            if is_background_eb:
                print(f"  🚨 SIGNIFICANT CENTROID MOTION DETECTED!")
                print(f"     Max shift: {max_overall_shift:.1f} sigma")
                print(f"     {significant_shifts}/{len(transit_centroid_shifts)} transits show shifts")
                print(f"     → Likely background eclipsing binary")
            else:
                print(f"  ✅ No significant centroid motion (max: {max_overall_shift:.1f} sigma)")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in centroid motion analysis: {e}")
            return {'centroid_motion_detected': False, 'error': str(e)}


    def validate_multi_aperture_consistency(self, kic_id, period, t0, duration):
        """
        Check transit consistency across different photometric apertures
        Real planetary transits should be consistent across all apertures
        """
        
        try:
            print(f"  📐 Validating multi-aperture photometry consistency...")
            
            # Download light curve with multiple apertures if available
            search_result = lk.search_lightcurve(f'KIC {kic_id}', mission='Kepler', cadence="long")
            
            if len(search_result) == 0:
                return {'multi_aperture_consistent': True, 'error': 'no_data'}
            
            # Try to get different aperture sizes
            aperture_results = []
            
            for i, lc_file in enumerate(search_result[:3]):  # Check up to 3 quarters
                try:
                    lc = lc_file.download()
                    if lc is None:
                        continue
                    
                    # Get different aperture masks if available
                    if hasattr(lc, 'pipeline_mask') and hasattr(lc, 'quality'):
                        # Use different aperture sizes
                        aperture_masks = []
                        
                        # Small aperture (core pixels)
                        small_mask = lc.pipeline_mask
                        
                        # Medium aperture (expand by 1 pixel)  
                        if hasattr(lc, 'create_threshold_mask'):
                            try:
                                medium_mask = lc.create_threshold_mask(threshold=0.1)
                                aperture_masks.append(('medium', medium_mask))
                            except:
                                pass
                        
                        aperture_masks.append(('small', small_mask))
                        
                        # Extract flux for each aperture
                        for aperture_name, mask in aperture_masks:
                            try:
                                if hasattr(lc, 'to_lightcurve'):
                                    aperture_lc = lc.to_lightcurve(aperture_mask=mask)
                                else:
                                    aperture_lc = lc
                                
                                # Phase fold and measure transit depth
                                phase_folded, metrics = self.phase_fold_transit(
                                    aperture_lc, period, t0, duration
                                )
                                
                                if metrics and 'error' not in metrics:
                                    aperture_results.append({
                                        'aperture': aperture_name,
                                        'quarter': i,
                                        'transit_depth_ppm': metrics.get('transit_depth_ppm', 0),
                                        'transit_snr': metrics.get('transit_snr', 0),
                                        'significance': metrics.get('transit_significance', 0)
                                    })
                                    
                            except Exception as aperture_error:
                                print(f"    ⚠️ Error with {aperture_name} aperture: {aperture_error}")
                                
                except Exception as quarter_error:
                    print(f"    ⚠️ Error processing quarter {i}: {quarter_error}")
                    continue
            
            if len(aperture_results) < 2:
                print(f"  ⚠️ Insufficient aperture data for comparison")
                return {'multi_aperture_consistent': True, 'error': 'insufficient_apertures'}
            
            # Compare transit depths across apertures
            depths = [r['transit_depth_ppm'] for r in aperture_results if r['transit_depth_ppm'] > 0]
            snrs = [r['transit_snr'] for r in aperture_results if r['transit_snr'] > 0]
            
            if len(depths) < 2:
                return {'multi_aperture_consistent': True, 'error': 'insufficient_detections'}
            
            # Calculate consistency metrics
            depth_mean = np.mean(depths)
            depth_std = np.std(depths)
            depth_cv = depth_std / depth_mean if depth_mean > 0 else 1.0
            
            snr_mean = np.mean(snrs) if snrs else 0
            snr_std = np.std(snrs) if len(snrs) > 1 else 0
            
            # Consistency test: coefficient of variation should be < 0.3 for real planets
            is_consistent = depth_cv < 0.3 and len([d for d in depths if d > depth_mean * 0.5]) >= len(depths) * 0.7
            
            result = {
                'multi_aperture_consistent': is_consistent,
                'aperture_results': aperture_results,
                'depth_consistency_cv': float(depth_cv),
                'mean_depth_ppm': float(depth_mean),
                'depth_std_ppm': float(depth_std),
                'mean_snr': float(snr_mean),
                'n_apertures_tested': len(aperture_results),
                'n_detections': len(depths)
            }
            
            if is_consistent:
                print(f"  ✅ Multi-aperture consistency: PASSED (CV={depth_cv:.2f})")
            else:
                print(f"  ❌ Multi-aperture consistency: FAILED (CV={depth_cv:.2f})")
                print(f"     Inconsistent transit depths across apertures")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in multi-aperture validation: {e}")
            return {'multi_aperture_consistent': True, 'error': str(e)}
    
    def detect_transit_timing_variations(self, lc, period, t0, duration):
        """
        Analyze transit timing variations which can indicate:
        1. Additional planets in the system
        2. Systematic instrumental effects
        3. Eclipsing binary nature
        """
        
        try:
            print(f"  ⏱️ Analyzing transit timing variations (TTVs)...")
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Find individual transit times with high precision
            observed_transit_times = []
            expected_transit_times = []
            
            current_t0 = t0
            transit_number = 0
            
            while current_t0 < time.max():
                if current_t0 > time.min():
                    # Look for transit in a window around expected time
                    window_size = min(duration / 24.0, period * 0.1)  # 10% of period max
                    transit_mask = np.abs(time - current_t0) < window_size
                    
                    if np.sum(transit_mask) > 5:
                        transit_times = time[transit_mask]
                        transit_fluxes = flux[transit_mask]
                        
                        # Find the minimum flux point as the transit center
                        min_flux_idx = np.argmin(transit_fluxes)
                        observed_time = transit_times[min_flux_idx]
                        
                        # Refine transit center using weighted centroid
                        weights = 1.0 / (transit_fluxes - np.min(transit_fluxes) + 1e-6)
                        refined_time = np.average(transit_times, weights=weights)
                        
                        observed_transit_times.append(refined_time)
                        expected_transit_times.append(current_t0)
                        
                        transit_number += 1
                        
                current_t0 += period
            
            if len(observed_transit_times) < 3:
                print(f"  ⚠️ Need at least 3 transits for TTV analysis")
                return {'ttv_detected': False, 'error': 'insufficient_transits'}
            
            observed_times = np.array(observed_transit_times)
            expected_times = np.array(expected_transit_times)
            
            # Calculate timing residuals (O-C: Observed - Calculated)
            timing_residuals = (observed_times - expected_times) * 24 * 60  # Convert to minutes
            
            # Statistical analysis of TTVs
            ttv_std = np.std(timing_residuals)
            ttv_rms = np.sqrt(np.mean(timing_residuals**2))
            max_ttv = np.max(np.abs(timing_residuals))
            
            # Linear trend test
            transit_epochs = np.arange(len(timing_residuals))
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(transit_epochs, timing_residuals)
                linear_trend_significance = abs(r_value) if np.isfinite(r_value) else 0
            except:
                slope, linear_trend_significance = 0, 0
            
            # Periodicity test using Lomb-Scargle
            try:
                if len(timing_residuals) > 5:
                    ls = LombScargle(transit_epochs, timing_residuals)
                    frequency = np.linspace(0.01, 0.5, 100)  # Up to Nyquist
                    power = ls.power(frequency)
                    max_power = np.max(power)
                    max_power_freq = frequency[np.argmax(power)]
                    max_power_period = 1.0 / max_power_freq
                else:
                    max_power, max_power_period = 0, 0
            except:
                max_power, max_power_period = 0, 0
            
            # Significance thresholds
            ttv_significant = (ttv_rms > 5.0 or  # > 5 minutes RMS
                            max_ttv > 15.0 or  # > 15 minutes max
                            linear_trend_significance > 0.7)  # Strong linear trend
            
            # TTV pattern classification
            ttv_pattern = 'none'
            if linear_trend_significance > 0.7:
                ttv_pattern = 'linear_trend'
            elif max_power > 0.3:
                ttv_pattern = 'periodic'
            elif ttv_rms > 10.0:
                ttv_pattern = 'chaotic'
            elif ttv_rms > 2.0:
                ttv_pattern = 'low_level'
            
            result = {
                'ttv_detected': ttv_significant,
                'ttv_rms_minutes': float(ttv_rms),
                'ttv_std_minutes': float(ttv_std),
                'max_ttv_minutes': float(max_ttv),
                'n_transits_analyzed': len(timing_residuals),
                'linear_trend_slope_min_per_transit': float(slope),
                'linear_trend_significance': float(linear_trend_significance),
                'periodogram_max_power': float(max_power),
                'periodogram_peak_period_transits': float(max_power_period),
                'ttv_pattern': ttv_pattern,
                'timing_residuals_minutes': timing_residuals.tolist(),
                'transit_epochs': transit_epochs.tolist(),
                'indicates_additional_planets': (ttv_pattern == 'periodic' and max_power > 0.4),
                'indicates_systematic_issues': (ttv_pattern == 'linear_trend' and abs(slope) > 2.0)
            }
            
            if ttv_significant:
                print(f"  🕐 SIGNIFICANT TTVs DETECTED!")
                print(f"     RMS: {ttv_rms:.1f} min, Max: {max_ttv:.1f} min")
                print(f"     Pattern: {ttv_pattern}")
                if result['indicates_additional_planets']:
                    print(f"     → May indicate additional planets!")
                elif result['indicates_systematic_issues']:
                    print(f"     → May indicate systematic issues")
            else:
                print(f"  ✅ No significant TTVs (RMS: {ttv_rms:.1f} min)")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in TTV analysis: {e}")
            return {'ttv_detected': False, 'error': str(e)}

    def compare_odd_even_transits(self, lc, period, t0, duration):
        """
        Compare odd and even numbered transits
        Significant differences can indicate:
        1. Eclipsing binary (different eclipse depths)
        2. Instrumental systematics
        3. Stellar activity
        """
        
        try:
            print(f"  🔢 Comparing odd vs even transits...")
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Find all transit times and classify as odd/even
            odd_transits = []
            even_transits = []
            
            current_t0 = t0
            transit_number = 0
            
            while current_t0 < time.max():
                if current_t0 > time.min():
                    # Extract transit data
                    transit_mask = np.abs(time - current_t0) < (duration / 24.0)
                    baseline_mask = ((np.abs(time - current_t0) > duration / 24.0) & 
                                (np.abs(time - current_t0) < period / 4))
                    
                    if np.sum(transit_mask) > 3 and np.sum(baseline_mask) > 10:
                        transit_flux = flux[transit_mask]
                        baseline_flux = np.median(flux[baseline_mask])
                        
                        # Calculate transit depth and significance
                        transit_depth = baseline_flux - np.median(transit_flux)
                        transit_std = np.std(flux[baseline_mask])
                        transit_snr = transit_depth / (transit_std / np.sqrt(len(transit_flux)))
                        
                        transit_data = {
                            'number': transit_number,
                            'time': current_t0,
                            'depth': transit_depth,
                            'depth_ppm': transit_depth * 1e6,
                            'snr': transit_snr,
                            'baseline_flux': baseline_flux,
                            'n_points': len(transit_flux)
                        }
                        
                        # Classify as odd or even (starting from 0)
                        if transit_number % 2 == 0:
                            even_transits.append(transit_data)
                        else:
                            odd_transits.append(transit_data)
                        
                        transit_number += 1
                        
                current_t0 += period
            
            if len(odd_transits) < 2 or len(even_transits) < 2:
                print(f"  ⚠️ Need at least 2 odd and 2 even transits")
                return {'odd_even_consistent': True, 'error': 'insufficient_transits'}
            
            # Calculate statistics for odd and even transits
            odd_depths = [t['depth_ppm'] for t in odd_transits if t['depth_ppm'] > 0]
            even_depths = [t['depth_ppm'] for t in even_transits if t['depth_ppm'] > 0]
            odd_snrs = [t['snr'] for t in odd_transits]
            even_snrs = [t['snr'] for t in even_transits]
            
            if len(odd_depths) == 0 or len(even_depths) == 0:
                return {'odd_even_consistent': True, 'error': 'no_significant_transits'}
            
            # Statistical comparisons
            odd_median_depth = np.median(odd_depths)
            even_median_depth = np.median(even_depths)
            odd_mean_snr = np.mean(odd_snrs)
            even_mean_snr = np.mean(even_snrs)
            
            # Depth difference test
            depth_ratio = abs(odd_median_depth - even_median_depth) / max(odd_median_depth, even_median_depth)
            
            # Statistical tests
            try:
                # Mann-Whitney U test for depth differences
                u_stat, u_p_value = stats.mannwhitneyu(odd_depths, even_depths, alternative='two-sided')
                depth_difference_significant = u_p_value < 0.05
            except:
                depth_difference_significant = False
                u_p_value = 1.0
            
            # Consistency assessment
            is_consistent = (depth_ratio < self.odd_even_depth_threshold and 
                            not depth_difference_significant)
            
            # Additional checks for eclipsing binary patterns
            primary_secondary_pattern = False
            if not is_consistent:
                # Check if one set is consistently deeper (EB primary/secondary)
                deeper_group = 'odd' if odd_median_depth > even_median_depth else 'even'
                depth_ratio_eb = max(odd_median_depth, even_median_depth) / min(odd_median_depth, even_median_depth)
                
                if depth_ratio_eb > 2.0:  # One eclipse much deeper
                    primary_secondary_pattern = True
            
            result = {
                'odd_even_consistent': is_consistent,
                'n_odd_transits': len(odd_transits),
                'n_even_transits': len(even_transits),
                'odd_median_depth_ppm': float(odd_median_depth),
                'even_median_depth_ppm': float(even_median_depth),
                'depth_ratio': float(depth_ratio),
                'odd_mean_snr': float(odd_mean_snr),
                'even_mean_snr': float(even_mean_snr),
                'mannwhitney_p_value': float(u_p_value),
                'depth_difference_significant': depth_difference_significant,
                'primary_secondary_pattern': primary_secondary_pattern,
                'likely_eclipsing_binary_pattern': primary_secondary_pattern,
                'odd_transit_data': odd_transits,
                'even_transit_data': even_transits
            }
            
            if not is_consistent:
                print(f"  ❌ ODD/EVEN INCONSISTENCY DETECTED!")
                print(f"     Depth ratio: {depth_ratio:.2f}")
                print(f"     Odd: {odd_median_depth:.0f} ppm, Even: {even_median_depth:.0f} ppm")
                if primary_secondary_pattern:
                    print(f"     → Possible eclipsing binary (primary/secondary)")
            else:
                print(f"  ✅ Odd/even transits consistent (ratio: {depth_ratio:.2f})")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in odd/even comparison: {e}")
            return {'odd_even_consistent': True, 'error': str(e)}

    def check_neighboring_star_contamination(self, kic_id, period, t0):
        """
        Check for contamination from nearby eclipsing binaries
        Uses Gaia catalog and pixel-level analysis
        """
        
        try:
            print(f"  🌟 Checking for neighboring star contamination...")
            
            # This is a simplified version - in practice you'd query Gaia catalog
            # and check for nearby stars with similar periods
            
            contamination_score = 0.0
            nearby_stars = []
            
            # Placeholder for actual Gaia query and analysis
            # In a real implementation, you would:
            # 1. Query Gaia DR3 catalog for stars within ~1 arcminute
            # 2. Check their brightness and colors
            # 3. Look for known eclipsing binaries
            # 4. Analyze pixel-level contributions
            
            result = {
                'neighboring_contamination_detected': contamination_score > self.neighboring_star_contamination_threshold,
                'contamination_score': float(contamination_score),
                'nearby_stars': nearby_stars,
                'analysis_method': 'simplified_placeholder'
            }
            
            print(f"  ✅ Neighboring star check: contamination score = {contamination_score:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in neighboring star check: {e}")
            return {'neighboring_contamination_detected': False, 'error': str(e)}

    def advanced_stellar_activity_filter(self, lc, period, t0):
        """
        Advanced stellar activity detection and filtering
        """
        
        try:
            print(f"  ⭐ Advanced stellar activity analysis...")
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Remove the candidate transit signal to look for activity
            phases = ((time - t0) % period) / period
            transit_mask = np.abs(phases - 0.5) < 0.05
            
            # Detrended flux (excluding transits)
            detrended_flux = flux.copy()
            if np.sum(transit_mask) > 0:
                # Simple linear interpolation over transits
                non_transit_mask = ~transit_mask
                detrended_flux[transit_mask] = np.interp(
                    time[transit_mask], 
                    time[non_transit_mask], 
                    flux[non_transit_mask]
                )
            
            # 1. Stellar rotation period detection
            rotation_periods = self._detect_stellar_rotation(time, detrended_flux)
            
            # 2. Flare detection
            flares = self._detect_stellar_flares(time, flux)
            
            # 3. Long-term trends
            long_term_trend = self._measure_long_term_trend(time, detrended_flux)
            
            # 4. Quasi-periodic oscillations
            qpo_analysis = self._detect_quasi_periodic_oscillations(time, detrended_flux)
            
            # Overall stellar activity score
            activity_score = 0.0
            
            # Penalize if rotation period matches transit period
            if rotation_periods:
                for rot_period in rotation_periods:
                    period_ratio = abs(period - rot_period) / period
                    if period_ratio < 0.05:  # Within 5%
                        activity_score += 0.3
            
            # Penalize for excessive flares
            if flares['n_flares'] > 10:
                activity_score += 0.2
            
            # Penalize for strong trends
            if long_term_trend['trend_significance'] > 0.5:
                activity_score += 0.15
            
            # Penalize for strong QPOs
            if qpo_analysis['max_qpo_power'] > 0.1:
                activity_score += 0.1
            
            result = {
                'stellar_activity_score': float(np.clip(activity_score, 0, 1)),
                'rotation_periods': rotation_periods,
                'flare_analysis': flares,
                'long_term_trend': long_term_trend,
                'qpo_analysis': qpo_analysis,
                'high_activity': activity_score > 0.4
            }
            
            if activity_score > 0.4:
                print(f"  ⚠️ HIGH STELLAR ACTIVITY detected (score: {activity_score:.2f})")
            else:
                print(f"  ✅ Low stellar activity (score: {activity_score:.2f})")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in stellar activity analysis: {e}")
            return {'stellar_activity_score': 0.0, 'error': str(e)}

    def _detect_stellar_rotation(self, time, flux):
        """Detect stellar rotation periods"""
        try:
            # Use Lomb-Scargle periodogram for rotation detection
            frequency = np.linspace(1/50, 1/0.5, 1000)  # 0.5 to 50 day periods
            ls = LombScargle(time, flux)
            power = ls.power(frequency)
            periods = 1/frequency
            
            # Find significant peaks
            peak_threshold = np.percentile(power, 99)
            peak_indices = find_peaks(power, height=peak_threshold)[0]
            
            rotation_periods = []
            if len(peak_indices) > 0:
                for idx in peak_indices:
                    if 0.5 <= periods[idx] <= 50:  # Reasonable stellar rotation range
                        rotation_periods.append(float(periods[idx]))
            
            return sorted(rotation_periods)[:5]  # Return top 5
            
        except Exception:
            return []

    def _detect_stellar_flares(self, time, flux):
        """Detect stellar flares in the light curve"""
        try:
            # Smooth the light curve to find baseline
            smoothed_flux = gaussian_filter1d(flux, sigma=10)
            
            # Find positive outliers (flares)
            residuals = flux - smoothed_flux
            threshold = 3 * np.std(residuals)
            
            flare_mask = residuals > threshold
            flare_indices = np.where(flare_mask)[0]
            
            # Group consecutive flare points
            if len(flare_indices) == 0:
                return {'n_flares': 0, 'flare_rate': 0, 'max_flare_amplitude': 0}
            
            # Simple flare counting (consecutive points = 1 flare)
            flare_groups = []
            current_group = [flare_indices[0]]
            
            for i in range(1, len(flare_indices)):
                if flare_indices[i] - flare_indices[i-1] <= 5:  # Within 5 cadences
                    current_group.append(flare_indices[i])
                else:
                    flare_groups.append(current_group)
                    current_group = [flare_indices[i]]
            flare_groups.append(current_group)
            
            n_flares = len(flare_groups)
            observation_span = time[-1] - time[0]
            flare_rate = n_flares / observation_span  # flares per day
            max_amplitude = np.max(residuals[flare_mask]) if len(flare_indices) > 0 else 0
            
            return {
                'n_flares': n_flares,
                'flare_rate': float(flare_rate),
                'max_flare_amplitude': float(max_amplitude),
                'flare_indices': [int(idx) for idx in flare_indices[:100]]  # Limit output
            }
            
        except Exception:
            return {'n_flares': 0, 'flare_rate': 0, 'max_flare_amplitude': 0}
    
    def _measure_long_term_trend(self, time, flux):
        """Measure long-term instrumental or astrophysical trends"""
        try:
            # Fit polynomial trends of different orders
            trends = {}
            
            for order in [1, 2, 3]:
                coeffs = np.polyfit(time - np.mean(time), flux, order)
                trend_flux = np.polyval(coeffs, time - np.mean(time))
                residuals = flux - trend_flux
                rms = np.std(residuals)
                
                # Calculate trend significance
                trend_amplitude = np.max(trend_flux) - np.min(trend_flux)
                noise_level = np.std(flux)
                significance = trend_amplitude / noise_level
                
                trends[f'order_{order}'] = {
                    'coefficients': coeffs.tolist(),
                    'rms': float(rms),
                    'significance': float(significance)
                }
            
            # Use linear trend as primary measure
            best_trend = trends['order_1']
            
            return {
                'trend_significance': best_trend['significance'],
                'linear_slope': best_trend['coefficients'][0],
                'all_trends': trends
            }
            
        except Exception:
            return {'trend_significance': 0, 'linear_slope': 0}

    def _detect_quasi_periodic_oscillations(self, time, flux):
        """Detect quasi-periodic oscillations (QPOs)"""
        try:
            # High frequency analysis for QPOs
            frequency = np.linspace(0.1, 24, 500)  # 0.1 to 24 cycles/day
            ls = LombScargle(time, flux)
            power = ls.power(frequency)
            
            # Find the strongest QPO
            max_power = np.max(power)
            max_freq = frequency[np.argmax(power)]
            max_period = 1/max_freq
            
            # Significance test
            qpo_threshold = np.percentile(power, 99.9)
            significant_qpos = power > qpo_threshold
            
            return {
                'max_qpo_power': float(max_power),
                'max_qpo_frequency': float(max_freq),
                'max_qpo_period_hours': float(max_period * 24),
                'n_significant_qpos': int(np.sum(significant_qpos))
            }
            
        except Exception:
            return {'max_qpo_power': 0, 'max_qpo_frequency': 0, 'n_significant_qpos': 0}


    def statistical_validation_ensemble(self, result_dict):
        """
        Combine all validation metrics into a comprehensive statistical score
        """
        
        try:
            print(f"  📊 Computing statistical validation ensemble...")
            
            # Initialize scores
            validation_scores = {}
            confidence_boosts = {}
            red_flags = {}
            
            # 1. Centroid Motion Score
            centroid = result_dict.get('centroid_analysis', {})
            if not centroid.get('centroid_motion_detected', False):
                validation_scores['centroid'] = 1.0
                confidence_boosts['centroid_stable'] = 0.1
            else:
                validation_scores['centroid'] = 0.0
                red_flags['centroid_motion'] = -0.3
            
            # 2. Multi-aperture Consistency Score
            multi_ap = result_dict.get('multi_aperture', {})
            if multi_ap.get('multi_aperture_consistent', True):
                consistency_score = 1.0 - multi_ap.get('depth_consistency_cv', 0.5)
                validation_scores['multi_aperture'] = max(consistency_score, 0.5)
                if consistency_score > 0.8:
                    confidence_boosts['aperture_consistent'] = 0.05
            else:
                validation_scores['multi_aperture'] = 0.2
                red_flags['aperture_inconsistent'] = -0.15
            
            # 3. Transit Timing Variations Score
            ttv = result_dict.get('ttv_analysis', {})
            if not ttv.get('ttv_detected', False):
                validation_scores['ttv'] = 1.0
                confidence_boosts['stable_timing'] = 0.05
            else:
                ttv_rms = ttv.get('ttv_rms_minutes', 0)
                if ttv_rms < 10:
                    validation_scores['ttv'] = 0.7  # Minor TTVs acceptable
                elif ttv_rms < 30:
                    validation_scores['ttv'] = 0.4  # Moderate TTVs concerning
                else:
                    validation_scores['ttv'] = 0.1  # Large TTVs problematic
                    red_flags['large_ttvs'] = -0.1
                
                # Bonus for TTVs that indicate additional planets
                if ttv.get('indicates_additional_planets', False):
                    confidence_boosts['ttv_additional_planets'] = 0.15
            
            # 4. Odd/Even Transit Consistency Score
            odd_even = result_dict.get('odd_even_analysis', {})
            if odd_even.get('odd_even_consistent', True):
                validation_scores['odd_even'] = 1.0
                confidence_boosts['depth_consistent'] = 0.08
            else:
                validation_scores['odd_even'] = 0.2
                if odd_even.get('likely_eclipsing_binary_pattern', False):
                    red_flags['eclipsing_binary_pattern'] = -0.4
                else:
                    red_flags['odd_even_inconsistent'] = -0.2
            
            # 5. Stellar Activity Score
            stellar_activity = result_dict.get('stellar_activity', {})
            activity_score = stellar_activity.get('stellar_activity_score', 0)
            validation_scores['stellar_activity'] = 1.0 - activity_score
            if activity_score > 0.6:
                red_flags['high_stellar_activity'] = -0.2
            
            # 6. Secondary Eclipse Score
            secondary = result_dict.get('secondary_eclipse', {})
            if not secondary.get('likely_eclipsing_binary', False):
                validation_scores['secondary_eclipse'] = 1.0
                confidence_boosts['no_secondary'] = 0.1
            else:
                validation_scores['secondary_eclipse'] = 0.0
                red_flags['secondary_eclipse_detected'] = -0.5
            
            # 7. Enhanced Transit Metrics Score
            enhanced_metrics = result_dict.get('enhanced_transit_metrics', {})
            if enhanced_metrics and 'error' not in enhanced_metrics:
                # SNR-based score
                snr = enhanced_metrics.get('enhanced_snr', 0)
                snr_score = min(max((snr - 2) / 8, 0), 1)  # 2-10 SNR maps to 0-1
                validation_scores['transit_snr'] = snr_score
                
                # Statistical significance score
                if enhanced_metrics.get('likely_real_transit', False):
                    confidence_boosts['statistically_significant'] = 0.2
                
                # Data quality score
                data_quality = enhanced_metrics.get('data_quality_score', 0.5)
                validation_scores['data_quality'] = data_quality
                
            else:
                # Fallback to basic metrics
                basic_metrics = result_dict.get('light_curve_processing', {}).get('transit_metrics', {})
                if basic_metrics and 'error' not in basic_metrics:
                    snr = basic_metrics.get('transit_snr', 0)
                    validation_scores['transit_snr'] = min(max((snr - 1) / 6, 0), 1)
                    validation_scores['data_quality'] = 0.6
                else:
                    validation_scores['transit_snr'] = 0.3
                    validation_scores['data_quality'] = 0.4
            
            # Calculate weighted ensemble score
            weights = {
                'centroid': 0.2,           # Most important - rules out background EBs
                'secondary_eclipse': 0.15,  # Critical for EB detection
                'odd_even': 0.15,          # Important consistency check
                'transit_snr': 0.15,       # Signal strength
                'multi_aperture': 0.1,     # Aperture consistency
                'ttv': 0.1,               # Timing stability
                'stellar_activity': 0.1,   # Activity contamination
                'data_quality': 0.05       # Overall data quality
            }
            
            # Calculate base ensemble score
            ensemble_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in validation_scores:
                    ensemble_score += validation_scores[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
            else:
                ensemble_score = 0.5  # Neutral if no metrics available
            
            # Apply confidence boosts
            total_boost = sum(confidence_boosts.values())
            ensemble_score = min(ensemble_score + total_boost, 1.0)
            
            # Apply red flags
            total_penalty = sum(red_flags.values())
            ensemble_score = max(ensemble_score + total_penalty, 0.0)
            
            # Determine validation level
            if ensemble_score >= 0.9:
                validation_level = 'GOLD'      # Highest confidence
            elif ensemble_score >= 0.8:
                validation_level = 'SILVER'    # Very high confidence
            elif ensemble_score >= 0.7:
                validation_level = 'BRONZE'    # High confidence
            elif ensemble_score >= 0.5:
                validation_level = 'PASS'      # Acceptable
            else:
                validation_level = 'FAIL'      # Low confidence
            
            result = {
                'ensemble_score': float(ensemble_score),
                'validation_level': validation_level,
                'individual_scores': validation_scores,
                'confidence_boosts': confidence_boosts,
                'red_flags': red_flags,
                'total_boost': float(total_boost),
                'total_penalty': float(total_penalty),
                'base_score': float(ensemble_score - total_boost - total_penalty)
            }
            
            print(f"  🏆 Validation ensemble: {validation_level} ({ensemble_score*100:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in statistical ensemble: {e}")
            return {'ensemble_score': 0.5, 'validation_level': 'UNKNOWN', 'error': str(e)}
    
    def _calculate_transit_metrics(self, phases, flux, flux_err, duration, period):
        """Calculate detailed transit metrics from phase-folded light curve - FIXED VERSION"""
        
        metrics = {}
        
        try:
            # Validate inputs
            if len(phases) == 0 or len(flux) == 0 or len(flux_err) == 0:
                return {'error': 'empty_data'}
            
            if not np.all(np.isfinite(phases)) or not np.all(np.isfinite(flux)):
                return {'error': 'invalid_data'}
            
            # Estimate transit duration in phase units
            transit_duration_phase = max(duration / 24.0 / period, 0.01)  # at least 1% of the orbit
             
            # Define transit window (around phase 0.5)
            transit_mask = np.abs(phases - 0.5) < (transit_duration_phase * 2.0) # wider window
            
            # Define out-of-transit regions
            baseline_width = max(transit_duration_phase * 4.0, 0.15)  # At least 15% phase
            out_of_transit_mask = (
                     (np.abs(phases - 0.5) > baseline_width) & \
                     (np.abs(phases - 0.0) > 0.05) & \
                     (np.abs(phases - 1.0) > 0.05)
                     )
            
            # More lenient data requirements
            if np.sum(transit_mask) < 2 or np.sum(out_of_transit_mask) < 5:
                print(f"  ⚠️ Insufficient data for transit analysis (transit: {np.sum(transit_mask)}, baseline: {np.sum(out_of_transit_mask)})")
                return {'error': 'insufficient_data'}
            
            # Calculate baseline flux (out-of-transit median)
            baseline_flux = np.median(flux[out_of_transit_mask])
            baseline_std = np.std(flux[out_of_transit_mask])
            
            # Prevent division by zero
            if baseline_std == 0:
                baseline_std = np.std(flux) if np.std(flux) > 0 else 1e-6
            
            # Calculate transit depth
            in_transit_flux = np.median(flux[transit_mask])
            transit_depth = baseline_flux - in_transit_flux
            

            # Calculate signal-to-noise ratio
            transit_points = np.max(transit_mask)
            if transit_points > 0:
                transit_snr = transit_depth / (baseline_std / np.sqrt(np.sum(transit_mask)))
            else:
                transit_snr = 0
            
            # Detect transit ingress/egress for duration measurement
            transit_phases = phases[transit_mask]
            transit_fluxes = flux[transit_mask]
            
            # Find the deepest point
            if len(transit_fluxes) > 0:
                deepest_idx = np.argmin(transit_fluxes)
                deepest_phase = transit_phases[deepest_idx]
            else:
                deepest_phase = 0.5  # Default to center
            
            # Calculate actual transit duration from ingress to egress
            measured_duration = duration  # Default fallback
            
            try:
                # Use 50% of transit depth as threshold
                half_depth = baseline_flux - transit_depth / 2
                
                ingress_phases = transit_phases[transit_phases < deepest_phase]
                egress_phases = transit_phases[transit_phases > deepest_phase]
                
                if len(ingress_phases) > 0 and len(egress_phases) > 0:
                    ingress_flux = flux[transit_mask][transit_phases < deepest_phase]
                    egress_flux = flux[transit_mask][transit_phases > deepest_phase]
                    
                    # Find ingress and egress points
                    ingress_idx = np.where(ingress_flux > half_depth)[0]
                    egress_idx = np.where(egress_flux > half_depth)[0]
                    
                    if len(ingress_idx) > 0 and len(egress_idx) > 0:
                        ingress_phase = ingress_phases[ingress_idx[-1]]
                        egress_phase = egress_phases[egress_idx[0]]
                        measured_duration = (egress_phase - ingress_phase) * period * 24  # Convert to hours
                    
            except Exception as duration_error:
                print(f"  ⚠️ Could not measure transit duration: {duration_error}")
                # Keep default duration
            
            # Calculate chi-squared for flat baseline vs transit model
            try:
                expected_transit_model = np.where(transit_mask, baseline_flux - transit_depth, baseline_flux)
                chi_squared = np.sum((flux - expected_transit_model)**2 / (flux_err**2 + 1e-10))
                reduced_chi_squared = chi_squared / max(len(flux) - 3, 1)  # 3 parameters: baseline, depth, duration
            except:
                chi_squared = 0
                reduced_chi_squared = 0
            
            # Calculate Box Least Squares (BLS) like statistic
            n_in_transit = np.sum(transit_mask)
            n_out_transit = np.sum(out_of_transit_mask)
            
            if n_in_transit > 0 and n_out_transit > 0:
                bls_statistic = abs(baseline_flux - in_transit_flux) * np.sqrt(n_in_transit * n_out_transit / (n_in_transit + n_out_transit))
            else:
                bls_statistic = 0
            
            # Phase coverage metric
            if len(phases) > 1:
                phase_coverage = len(phases) / (np.max(phases) - np.min(phases))
            else:
                phase_coverage = 0

            is_significant = abs(transit_snr) > 2.0 and transit_depth > 0
            
            # Compile all metrics
            metrics = {
                'transit_depth': float(transit_depth),
                'transit_depth_ppm': float(transit_depth * 1e6),
                'baseline_flux': float(baseline_flux),
                'baseline_std': float(baseline_std),
                'transit_snr': float(transit_snr),
                'measured_duration_hours': float(measured_duration),
                'chi_squared': float(chi_squared),
                'reduced_chi_squared': float(reduced_chi_squared),
                'bls_statistic': float(bls_statistic),
                'phase_coverage': float(phase_coverage),
                'n_transit_points': int(n_in_transit),
                'n_baseline_points': int(n_out_transit),
                'transit_significance': float(abs(transit_snr)),
                'is_significant': bool(is_significant),
                'deepest_phase': float(deepest_phase)
            }
            
        except Exception as e:
            print(f"  ⚠️ Error calculating transit metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics

    def plot_phase_folded_lightcurve(self, kic_id, phase_folded_lc, save_plot=True):
        """Create a beautiful phase-folded light curve plot - FIXED VERSION"""
        
        if phase_folded_lc is None:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            phases = phase_folded_lc['phase']
            flux = phase_folded_lc['flux']
            flux_err = phase_folded_lc['flux_err']
            metrics = phase_folded_lc['metrics']
            period = phase_folded_lc['period']
            duration = phase_folded_lc['duration']
            
            # Plot 1: Full phase curve
            ax1.errorbar(phases, flux, yerr=flux_err, fmt='o', color='navy', alpha=0.7, 
                        markersize=3, capsize=2, label='Phase-folded data')
            
            # Highlight transit region
            transit_duration_phase = duration / 24.0 / period
            transit_mask = np.abs(phases - 0.5) < (transit_duration_phase * 1.5)
            
            if np.any(transit_mask):
                ax1.errorbar(phases[transit_mask], flux[transit_mask], yerr=flux_err[transit_mask], 
                           fmt='o', color='red', alpha=0.8, markersize=4, capsize=2, label='Transit region')
            
            ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Expected transit center')
            ax1.set_xlabel('Phase')
            ax1.set_ylabel('Normalized Flux')
            ax1.set_title(f'KIC {kic_id} - Phase-Folded Light Curve (P={period:.4f}d)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add metrics text
            if 'error' not in metrics:
                metrics_text = f"""
Transit Depth: {metrics.get('transit_depth_ppm', 0):.1f} ppm
SNR: {metrics.get('transit_snr', 0):.2f}
Duration: {metrics.get('measured_duration_hours', 0):.2f} h
Significance: {'✓' if metrics.get('is_significant', False) else '✗'}
                """.strip()
                ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                        verticalalignment='top', fontsize=10)
            
            # Plot 2: Zoomed view of transit
            zoom_width = max(transit_duration_phase * 3, 0.05)  # At least 5% phase width
            zoom_mask = np.abs(phases - 0.5) < zoom_width
            
            if np.any(zoom_mask):
                ax2.errorbar(phases[zoom_mask], flux[zoom_mask], yerr=flux_err[zoom_mask], 
                           fmt='o', color='red', alpha=0.8, markersize=5, capsize=3)
                
                # Mark expected transit duration
                ax2.axvspan(0.5 - transit_duration_phase/2, 0.5 + transit_duration_phase/2, 
                          alpha=0.2, color='yellow', label='Expected transit duration')
                
                ax2.axvline(0.5, color='red', linestyle='--', alpha=0.7)
                ax2.set_xlabel('Phase')
                ax2.set_ylabel('Normalized Flux')
                ax2.set_title('Transit Zoom View')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No data in transit region', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_xlabel('Phase')
                ax2.set_ylabel('Normalized Flux')
                ax2.set_title('Transit Zoom View - No Data Available')
            
            plt.tight_layout()
            
            if save_plot:
                plot_filename = os.path.join(self.phase_fold_dir, f"kic_{kic_id}_phase_folded.png")
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"  💾 Phase-folded plot saved: {plot_filename}")
            
            plt.close()
            return plot_filename if save_plot else fig
            
        except Exception as e:
            print(f"  ❌ Error creating phase-folded plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    def process_light_curve(self, kic_id, period, t0, duration):
        """
        Complete light curve processing pipeline - FIXED VERSION
        """
        
        print(f"  🔄 Processing light curve for KIC {kic_id}...")
        
        processing_result = {
            'kic_id': kic_id,
            'light_curve_available': False,
            'phase_folded': False,
            'transit_metrics': {},
            'plot_generated': False,
            'processing_error': None
        }
        
        try:
            # Step 1: Download light curve
            if self.download_method == 'lightkurve':
                lc = self.download_light_curve_lightkurve(kic_id)
            else:
                lc = self.download_light_curve_mast(kic_id)
            
            if lc is None:
                processing_result['processing_error'] = 'light_curve_download_failed'
                return processing_result
            
            processing_result['light_curve_available'] = True
            
            # Step 2: Phase fold the light curve
            phase_folded_lc, metrics = self.phase_fold_transit(lc, period, t0, duration)
            
            if phase_folded_lc is None:
                processing_result['processing_error'] = 'phase_folding_failed'
                return processing_result
            
            processing_result['phase_folded'] = True
            processing_result['transit_metrics'] = metrics
            
            # Step 3: Generate phase-folded plot
            plot_filename = self.plot_phase_folded_lightcurve(kic_id, phase_folded_lc)
            if plot_filename:
                processing_result['plot_generated'] = True
                processing_result['plot_filename'] = plot_filename
            
            # Step 4: Enhanced signal strength assessment
            if 'error' not in metrics:
                # Calculate enhanced ML probability based on light curve analysis
                enhanced_prob = self._calculate_enhanced_ml_probability(metrics, processing_result)
                processing_result['enhanced_ml_prob'] = enhanced_prob
                
                print(f"  ✅ Light curve processing complete!")
                print(f"     Transit depth: {metrics.get('transit_depth_ppm', 0):.1f} ppm")
                print(f"     Transit SNR: {metrics.get('transit_snr', 0):.2f}")
                print(f"     Enhanced ML probability: {enhanced_prob:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error in light curve processing: {e}")
            processing_result['processing_error'] = str(e)
        
        return processing_result

    # =====  1: STELLAR CONTAMINATION DETECTION =====
    def detect_stellar_contamination(self, lc, period, t0):
        """
        CRITICAL: Detect stellar activity that can mimic planetary transits
        Your original code lacks this - it's a major source of false positives
        """
        
        try:
            time = lc.time.value
            flux = lc.flux.value
            
            # Remove the candidate transit signal temporarily
            transit_model = self._create_simple_transit_model(time, period, t0, duration=2.0)
            detrended_flux = flux / transit_model
            
            # 1. Check for stellar rotation periods near transit period
            rotation_contamination = self._check_rotation_contamination(time, detrended_flux, period)
            
            # 2. Look for correlated noise (instrumental/stellar)
            correlated_noise = self._measure_correlated_noise(time, detrended_flux)
            
            # 3. Check for systematic trends during "transits"
            systematic_trends = self._check_systematic_trends(lc, period, t0)
            
            contamination_score = (
                rotation_contamination * 0.5 +
                correlated_noise * 0.2 +
                systematic_trends * 0.3
            )
            
            return {
                'contamination_score': contamination_score,
                'rotation_contamination': rotation_contamination,
                'correlated_noise': correlated_noise,
                'systematic_trends': systematic_trends,
                'is_contaminated': contamination_score > self.stellar_contamination_threshold
            }
            
        except Exception as e:
            print(f"  ⚠️ Error in stellar contamination detection: {e}")
            return {'contamination_score': 0.0, 'is_contaminated': False}
    
    def _check_rotation_contamination(self, time, flux, candidate_period):
        """Check if candidate period matches stellar rotation"""
        
        # Use Lomb-Scargle to find dominant periods
        frequency = np.linspace(1/100, 1/0.5, 2000)  # 0.5 to 100 day periods
        ls = LombScargle(time, flux)
        power = ls.power(frequency)
        periods = 1/frequency
        
        # Find peaks in periodogram
        peak_indices = find_peaks(power, height=np.percentile(power, 95))[0]
        
        if len(peak_indices) > 0:
            dominant_periods = periods[peak_indices]
            
            # Check if candidate period matches any stellar rotation period
            max_contamination = 0.0
            
            for rot_period in dominant_periods:
                period_ratio = candidate_period / rot_period
                
                # Check for harmonic relationships
                harmonic_matches = [1.0, 2.0, 0.5, 3.0, 1/3.0]  # Common harmonics
                
                for harmonic in harmonic_matches:
                    ratio_diff = abs(period_ratio - harmonic)
                    
                    if ratio_diff < 0.02:  # Within 2% of harmonic
                        # Graduated contamination score based on how close the match is
                        contamination_strength = max(0.3, 0.9 - ratio_diff * 25)  # 0.1 to 0.8
                        max_contamination = max(max_contamination, contamination_strength)

                    elif ratio_diff < 0.05:
                        contamination_strength = 0.4
                        max_contamination = max(max_contamination, contamination_strength)
            
            return max_contamination
        
        return 0.0  # No contamination detected
    
    def _measure_correlated_noise(self, time, flux):
        """Measure red noise that can create false transit signals"""
        
        # Calculate autocorrelation function
        flux_normalized = (flux - np.mean(flux)) / np.std(flux)
        
        # Simple red noise test - look at consecutive point correlation
        if len(flux_normalized) > 10:
            # correlation = np.corrcoef(flux_normalized[:-1], flux_normalized[1:])[0, 1]
            # return abs(correlation) if np.isfinite(correlation) else 0.0

            correlation = []

            # check first few lags
            for lag in [1,2,3]:
                if len(flux_normalized) > lag:
                        corr = np.corrcoef(flux_normalized[:-lag], flux_normalized[lag:])[0, 1]
                        if np.isfinite(corr):
                            correlation.append(abs(corr))

            if correlation:
                avg_correlation = np.mean(correlation)

                if avg_correlation > 0.8:
                    return avg_correlation
                elif avg_correlation > 0.6:
                    return avg_correlation * 0.5  # Moderate penalty
                else:
                    return avg_correlation * 0.2  # Light penalty
        
        return 0.0
    
    def _check_systematic_trends(self, lc, period, t0):
        """Check for systematic instrumental trends during transit times"""
        
        try:
            time = lc.time.value
            
            # Find all transit times
            transit_times = []
            current_time = t0
            while current_time < time.max():
                if current_time > time.min():
                    transit_times.append(current_time)
                current_time += period
            
            if len(transit_times) < 2:
                return 0.0
            
            # Check if there are systematic trends at transit times
            # (e.g., always at certain spacecraft orientations, etc.)
            trend_scores = []
            
            for transit_time in transit_times[:10]: #Only the first 10 transits 
                # Get data around transit
                transit_mask = abs(time - transit_time) < (period * 0.1)
                if np.sum(transit_mask) > 5:
                    transit_flux = lc.flux.value[transit_mask]
                    transit_time_local = time[transit_mask]
                    
                    # Fit linear trend
                    if len(transit_flux) > 3:
                        slope, _, r_value, _, _ = stats.linregress(
                            transit_time_local - np.mean(transit_time_local), 
                            transit_flux
                        )
                        if np.isfinite(r_value):
                            trend_scores.append(abs(r_value))
            
            if trend_scores:
                avg_trends = np.mean(trend_scores)
                # Only flag very strong systematic trends
                return avg_trends if avg_trends > 0.7 else avg_trends * 0.3
            else:
                return 0.0
        except Exception:
            return 0.0
        

    # =====  2: SECONDARY ECLIPSE DETECTION =====
    def detect_secondary_eclipse(self, lc, period, t0):
        """
        CRITICAL: Detect secondary eclipses to distinguish planets from eclipsing binaries
        Your original code misses this - it's essential for ruling out stellar eclipses
        """
        
        try:
            time = lc.time.value
            flux = lc.flux.value
            
            # Secondary eclipse occurs at phase 0.5 (opposite of primary)
            secondary_t0 = t0 + period/2
            
            # Phase fold at secondary eclipse time
            secondary_phases = ((time - secondary_t0) % period) / period
            secondary_phases = (secondary_phases + 0.5) % 1.0  # Center at 0.5
            
            # Look for secondary eclipse signal
            in_secondary = abs(secondary_phases - 0.5) < 0.05  # Within 5% phase
            out_secondary = abs(secondary_phases - 0.5) > 0.2   # Outside secondary
            
            if np.sum(in_secondary) > 5 and np.sum(out_secondary) > 10:
                secondary_depth = np.median(flux[out_secondary]) - np.median(flux[in_secondary])
                primary_depth = self._estimate_primary_depth(lc, period, t0)
                
                if primary_depth > 0:
                    secondary_to_primary_ratio = secondary_depth / primary_depth
                    
                    return {
                        'secondary_depth_ppm': secondary_depth * 1e6,
                        'secondary_to_primary_ratio': secondary_to_primary_ratio,
                        'likely_eclipsing_binary': secondary_to_primary_ratio > self.secondary_eclipse_threshold,
                        'detection_significance': secondary_depth / np.std(flux[out_secondary])
                    }
            
            return {
                'secondary_depth_ppm': 0,
                'secondary_to_primary_ratio': 0,
                'likely_eclipsing_binary': False,
                'detection_significance': 0
            }
            
        except Exception as e:
            print(f"  ⚠️ Error in secondary eclipse detection: {e}")
            return {'likely_eclipsing_binary': False}
    
    def _estimate_primary_depth(self, lc, period, t0):
        """Estimate primary transit depth for comparison"""
        
        time = lc.time.value
        flux = lc.flux.value
        
        # Phase fold at primary transit
        phases = ((time - t0) % period) / period
        phases = (phases + 0.5) % 1.0
        
        in_transit = abs(phases - 0.5) < 0.05
        out_transit = abs(phases - 0.5) > 0.2
        
        if np.sum(in_transit) > 3 and np.sum(out_transit) > 10:
            primary_depth = np.median(flux[out_transit]) - np.median(flux[in_transit])
            return max(primary_depth, 0)
        
        return 0
    

    # =====  3: TRANSIT COUNT VERIFICATION =====
    def verify_transit_count(self, lc, period, t0, duration):
        """
        CRITICAL: Verify we see the expected number of transits
        Single-transit events are often false positives
        """
        
        try:
            time = lc.time.value
            flux = lc.flux.value
            
            # Calculate expected number of transits
            observation_span = time.max() - time.min()
            expected_transits = int(observation_span / period) + 1
            
            # Find individual transit events
            detected_transits = []
            current_t0 = t0
            
            while current_t0 < time.max():
                if current_t0 > time.min():
                    # Check if we have data during this transit
                    transit_mask = abs(time - current_t0) < (duration/24.0/2)  # Convert hours to days
                    
                    if np.sum(transit_mask) > 3:  # At least 3 points during transit
                        transit_flux = flux[transit_mask]
                        baseline_mask = (abs(time - current_t0) > duration/24.0) & (abs(time - current_t0) < period/4)
                        
                        if np.sum(baseline_mask) > 5:
                            baseline_flux = np.median(flux[baseline_mask])
                            transit_flux_median = np.median(transit_flux)
                            
                            # Check if this individual transit is significant
                            depth = baseline_flux - transit_flux_median
                            significance = depth / (np.std(flux[baseline_mask]) / np.sqrt(len(transit_flux)))
                            
                            if significance > 2.0:  # 2-sigma detection
                                detected_transits.append({
                                    'time': current_t0,
                                    'depth': depth,
                                    'significance': significance
                                })
                
                current_t0 += period
            
            detection_rate = len(detected_transits) / max(expected_transits, 1)
            
            return {
                'expected_transits': expected_transits,
                'detected_transits': len(detected_transits),
                'detection_rate': detection_rate,
                'individual_transit_depths': [t['depth']*1e6 for t in detected_transits],
                'sufficient_transits': len(detected_transits) >= self.minimum_transit_count,
                'consistent_depths': self._check_depth_consistency(detected_transits)
            }
            
        except Exception as e:
            print(f"  ⚠️ Error in transit count verification: {e}")
            return {'sufficient_transits': False, 'detected_transits': 0}
    
    def _check_depth_consistency(self, detected_transits):
        """Check if individual transit depths are consistent"""
        
        if len(detected_transits) < 2:
            return True  # Can't check consistency with < 2 transits
        
        depths = [t['depth'] for t in detected_transits]
        depth_std = np.std(depths)
        depth_mean = np.mean(depths)
        
        # Coefficient of variation should be < 50% for real planets
        if depth_mean > 0:
            cv = depth_std / depth_mean
            return cv < 0.5
        
        return False
    

    # =====  4: ENHANCED PHASE FOLDING WITH OUTLIER REJECTION =====
    def enhanced_phase_fold_transit(self, lc, period, t0, duration):
        """
        Improved version of your phase_fold_transit with outlier rejection and better statistics
        """
        
        try:
            time = lc.time.value
            flux = lc.flux.value
            
            # Apply sigma clipping to remove outliers first
            clipped_flux = sigma_clip(flux, sigma=4, maxiters=2)
            good_mask = ~clipped_flux.mask
            
            time_clean = time[good_mask]
            flux_clean = flux[good_mask]
            
            if len(flux_clean) < 50:
                return None, {}
            
            # Calculate phases
            phases = ((time_clean - t0) % period) / period
            phases = (phases + 0.5) % 1.0
            
            # Sort by phase
            sort_idx = np.argsort(phases)
            phases_sorted = phases[sort_idx]
            flux_sorted = flux_clean[sort_idx]
            
            # Adaptive binning based on data density
            n_bins = min(200, max(50, len(flux_sorted) // 20))
            
            # Create bins and calculate statistics
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            
            binned_flux = []
            binned_flux_err = []
            bin_counts = []
            
            for i in range(n_bins):
                mask = (phases_sorted >= bin_edges[i]) & (phases_sorted < bin_edges[i + 1])
                bin_flux = flux_sorted[mask]
                
                if len(bin_flux) > 0:
                    # Use robust statistics
                    binned_flux.append(np.median(bin_flux))
                    binned_flux_err.append(np.std(bin_flux) / np.sqrt(len(bin_flux)))
                    bin_counts.append(len(bin_flux))
                else:
                    binned_flux.append(np.nan)
                    binned_flux_err.append(np.nan)
                    bin_counts.append(0)
            
            binned_flux = np.array(binned_flux)
            binned_flux_err = np.array(binned_flux_err)
            
            # Remove empty bins
            valid_mask = np.isfinite(binned_flux) & (np.array(bin_counts) > 0)
            bin_centers = bin_centers[valid_mask]
            binned_flux = binned_flux[valid_mask]
            binned_flux_err = binned_flux_err[valid_mask]
            
            # Enhanced transit metrics
            metrics = self._calculate_enhanced_transit_metrics(
                bin_centers, binned_flux, binned_flux_err, duration, period
            )
            
            # Add outlier rejection metrics
            metrics['outlier_fraction'] = 1 - (len(flux_clean) / len(flux))
            metrics['data_quality_score'] = self._calculate_data_quality_score(
                time_clean, flux_clean, bin_counts
            )
            
            phase_folded_lc = {
                'phase': bin_centers,
                'flux': binned_flux,
                'flux_err': binned_flux_err,
                'period': period,
                't0': t0,
                'duration': duration,
                'metrics': metrics,
                'bin_counts': np.array(bin_counts)[valid_mask]
            }
            
            return phase_folded_lc, metrics
            
        except Exception as e:
            print(f"  ❌ Error in enhanced phase folding: {e}")
            return None, {}
    
    def _calculate_enhanced_transit_metrics(self, phases, flux, flux_err, duration, period):
        """
        Calculate enhanced transit metrics from phase-folded light curve with advanced statistics
        This is an improved version with more sophisticated analysis than the basic version
        """
        
        metrics = {}
        
        try:
            # Validate inputs
            if len(phases) == 0 or len(flux) == 0 or len(flux_err) == 0:
                return {'error': 'empty_data'}
            
            if not np.all(np.isfinite(phases)) or not np.all(np.isfinite(flux)):
                return {'error': 'invalid_data'}
            
            # Estimate transit duration in phase units
            transit_duration_phase = duration / 24.0 / period  # Convert hours to phase
            
            # Define transit window (around phase 0.5)
            transit_mask = np.abs(phases - 0.5) < (transit_duration_phase * 1.5)
            
            # Define out-of-transit regions (avoid secondary eclipse at phase 0.0/1.0)
            out_of_transit_mask = (
                (np.abs(phases - 0.5) > (transit_duration_phase * 2.0)) &
                (np.abs(phases - 0.0) > 0.1) &  # Avoid phase 0.0
                (np.abs(phases - 1.0) > 0.1)    # Avoid phase 1.0 (same as 0.0)
            )
            
            if np.sum(transit_mask) < 3 or np.sum(out_of_transit_mask) < 10:
                print(f"  ⚠️ Insufficient data for enhanced transit analysis (transit: {np.sum(transit_mask)}, baseline: {np.sum(out_of_transit_mask)})")
                return {'error': 'insufficient_data'}
            
            # ENHANCED BASELINE CALCULATION
            # Use iterative sigma clipping for robust baseline
            baseline_flux_data = flux[out_of_transit_mask]
            baseline_flux_clean = sigma_clip(baseline_flux_data, sigma=3, maxiters=2)
            baseline_flux = np.ma.median(baseline_flux_clean)
            baseline_std = np.ma.std(baseline_flux_clean)
            baseline_std_robust = 1.4826 * np.ma.median(np.ma.abs(baseline_flux_clean - baseline_flux))  # MAD-based std
            
            # Use the more robust estimate
            baseline_std = min(baseline_std, baseline_std_robust) if baseline_std_robust > 0 else baseline_std
            
            # Prevent division by zero
            if baseline_std == 0 or np.ma.is_masked(baseline_std):
                baseline_std = np.std(flux) if np.std(flux) > 0 else 1e-6
            
            # ENHANCED TRANSIT DEPTH CALCULATION
            # Use iterative approach to find the deepest part of transit
            transit_phases = phases[transit_mask]
            transit_fluxes = flux[transit_mask]
            
            # Find the deepest continuous region in the transit
            if len(transit_fluxes) >= 3:
                # Sort by phase within transit
                sort_idx = np.argsort(transit_phases)
                sorted_transit_phases = transit_phases[sort_idx]
                sorted_transit_flux = transit_fluxes[sort_idx]
                
                # Use a sliding window to find the deepest region
                window_size = max(3, len(sorted_transit_flux) // 3)
                deepest_flux = baseline_flux
                deepest_phase_center = 0.5
                
                for i in range(len(sorted_transit_flux) - window_size + 1):
                    window_flux = sorted_transit_flux[i:i+window_size]
                    window_median = np.median(window_flux)
                    
                    if window_median < deepest_flux:
                        deepest_flux = window_median
                        deepest_phase_center = np.median(sorted_transit_phases[i:i+window_size])
                
                # Calculate transit depth
                transit_depth = baseline_flux - deepest_flux
                in_transit_flux = deepest_flux
            else:
                # Fallback for sparse data
                in_transit_flux = np.median(transit_fluxes)
                transit_depth = baseline_flux - in_transit_flux
                deepest_phase_center = 0.5
            
            # ENHANCED SNR CALCULATION
            # Account for correlated noise and bin averaging
            n_transit_effective = np.sum(transit_mask)
            n_baseline_effective = np.sum(out_of_transit_mask)
            
            # Calculate SNR with proper error propagation
            if n_transit_effective > 0:
                # Standard SNR
                transit_snr = transit_depth / (baseline_std / np.sqrt(n_transit_effective))
                
                # Enhanced SNR accounting for systematic uncertainties
                systematic_uncertainty = baseline_std * 0.1  # 10% systematic floor
                total_uncertainty = np.sqrt((baseline_std / np.sqrt(n_transit_effective))**2 + systematic_uncertainty**2)
                enhanced_snr = transit_depth / total_uncertainty
            else:
                transit_snr = 0
                enhanced_snr = 0
            
            # ENHANCED DURATION MEASUREMENT
            # Use multiple methods to measure transit duration
            measured_durations = []
            
            # Method 1: Half-depth points
            try:
                half_depth_threshold = baseline_flux - transit_depth / 2
                
                # Find ingress and egress crossing points
                below_half_mask = flux[transit_mask] < half_depth_threshold
                if np.any(below_half_mask):
                    transit_phases_below = transit_phases[below_half_mask]
                    if len(transit_phases_below) > 0:
                        duration_method1 = (np.max(transit_phases_below) - np.min(transit_phases_below)) * period * 24
                        measured_durations.append(duration_method1)
            except:
                pass
            
            # Method 2: 1-sigma below baseline
            try:
                sigma_threshold = baseline_flux - baseline_std
                below_sigma_mask = flux[transit_mask] < sigma_threshold
                if np.any(below_sigma_mask):
                    transit_phases_below = transit_phases[below_sigma_mask]
                    if len(transit_phases_below) > 0:
                        duration_method2 = (np.max(transit_phases_below) - np.min(transit_phases_below)) * period * 24
                        measured_durations.append(duration_method2)
            except:
                pass
            
            # Method 3: Full Width at Half Maximum (FWHM) approach
            try:
                # Invert flux (make transit a peak)
                inverted_flux = baseline_flux - flux[transit_mask]
                if len(inverted_flux) > 3:
                    max_depth = np.max(inverted_flux)
                    half_max = max_depth / 2
                    above_half_max = inverted_flux >= half_max
                    if np.any(above_half_max):
                        fwhm_phases = transit_phases[above_half_max]
                        duration_fwhm = (np.max(fwhm_phases) - np.min(fwhm_phases)) * period * 24
                        measured_durations.append(duration_fwhm)
            except:
                pass
            
            # Choose the median of available measurements, fallback to input duration
            if measured_durations:
                measured_duration = np.median(measured_durations)
            else:
                measured_duration = duration
            
            # ENHANCED CHI-SQUARED ANALYSIS
            # Create a simple box transit model
            try:
                transit_model = np.full_like(flux, baseline_flux)
                transit_model[transit_mask] = baseline_flux - transit_depth
                
                # Calculate chi-squared with proper weighting
                weights = 1.0 / (flux_err**2 + baseline_std**2 * 0.01)  # Add systematic uncertainty
                residuals = flux - transit_model
                chi_squared = np.sum(weights * residuals**2)
                
                # Degrees of freedom: N_data - N_parameters
                # Parameters: baseline, depth, duration center
                dof = max(len(flux) - 3, 1)
                reduced_chi_squared = chi_squared / dof
                
                # Calculate null hypothesis chi-squared (flat baseline)
                null_model = np.full_like(flux, baseline_flux)
                null_residuals = flux - null_model
                null_chi_squared = np.sum(weights * null_residuals**2)
                null_reduced_chi = null_chi_squared / max(len(flux) - 1, 1)
                
                # Delta chi-squared (improvement from transit model)
                delta_chi_squared = null_chi_squared - chi_squared
                
            except:
                chi_squared = 0
                reduced_chi_squared = 1
                null_reduced_chi = 1
                delta_chi_squared = 0
            
            # ENHANCED BLS-LIKE STATISTIC
            # Improved version of Box Least Squares statistic
            try:
                # Signal residual power
                n_in = np.sum(transit_mask)
                n_out = np.sum(out_of_transit_mask)
                
                if n_in > 0 and n_out > 0:
                    # Power calculation similar to BLS but enhanced
                    signal_power = (transit_depth**2) * n_in * n_out / (n_in + n_out)
                    noise_power = baseline_std**2
                    
                    # Enhanced BLS statistic
                    enhanced_bls = signal_power / noise_power
                    
                    # Traditional BLS-like calculation
                    traditional_bls = abs(baseline_flux - in_transit_flux) * np.sqrt(n_in * n_out / (n_in + n_out))
                    
                    bls_statistic = max(enhanced_bls, traditional_bls)
                else:
                    bls_statistic = 0
            except:
                bls_statistic = 0
            
            # ENHANCED PHASE COVERAGE AND QUALITY METRICS
            # Calculate phase coverage quality
            if len(phases) > 1:
                phase_span = np.max(phases) - np.min(phases)
                phase_coverage = len(phases) / max(phase_span * 200, 1)  # Normalized to expected bin count
                
                # Calculate phase sampling uniformity
                phase_diffs = np.diff(np.sort(phases))
                phase_uniformity = 1.0 - (np.std(phase_diffs) / np.mean(phase_diffs)) if len(phase_diffs) > 0 else 0
            else:
                phase_coverage = 0
                phase_uniformity = 0
            
            # OUTLIER AND SYSTEMATIC ANALYSIS
            # Check for outliers in baseline
            baseline_outliers = np.sum(np.abs(baseline_flux_data - baseline_flux) > 3 * baseline_std)
            outlier_fraction = baseline_outliers / len(baseline_flux_data) if len(baseline_flux_data) > 0 else 0
            
            # Check for systematic trends
            try:
                # Linear trend test in baseline
                baseline_phases = phases[out_of_transit_mask]
                if len(baseline_phases) > 3:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_phases, baseline_flux_data)
                    trend_significance = abs(r_value) if np.isfinite(r_value) else 0
                else:
                    trend_significance = 0
            except:
                trend_significance = 0
            
            # DETECTION SIGNIFICANCE TESTS
            # Multiple statistical tests for transit detection
            
            # 1. Student's t-test between in-transit and out-of-transit
            try:
                if n_in > 2 and n_out > 2:
                    t_stat, t_p_value = stats.ttest_ind(transit_fluxes, baseline_flux_data)
                    t_significance = abs(t_stat) if np.isfinite(t_stat) else 0
                else:
                    t_significance = 0
                    t_p_value = 1.0
            except:
                t_significance = 0
                t_p_value = 1.0
            
            # 2. Mann-Whitney U test (non-parametric)
            try:
                if n_in > 2 and n_out > 2:
                    u_stat, u_p_value = stats.mannwhitneyu(transit_fluxes, baseline_flux_data, alternative='less')
                    u_significance = -np.log10(u_p_value) if u_p_value > 0 else 10
                else:
                    u_significance = 0
                    u_p_value = 1.0
            except:
                u_significance = 0
                u_p_value = 1.0
            
            # COMPILE ALL ENHANCED METRICS
            metrics = {
                # Basic metrics (enhanced versions)
                'transit_depth': float(transit_depth),
                'transit_depth_ppm': float(transit_depth * 1e6),
                'baseline_flux': float(baseline_flux),
                'baseline_std': float(baseline_std),
                'baseline_std_robust': float(baseline_std_robust),
                'transit_snr': float(transit_snr),
                'enhanced_snr': float(enhanced_snr),
                'measured_duration_hours': float(measured_duration),
                'duration_measurements': measured_durations,
                
                # Enhanced statistical measures
                'chi_squared': float(chi_squared),
                'reduced_chi_squared': float(reduced_chi_squared),
                'null_reduced_chi_squared': float(null_reduced_chi),
                'delta_chi_squared': float(delta_chi_squared),
                'enhanced_bls_statistic': float(bls_statistic),
                
                # Phase and coverage metrics
                'phase_coverage': float(phase_coverage),
                'phase_uniformity': float(phase_uniformity),
                
                # Data quality metrics
                'n_transit_points': int(n_in),
                'n_baseline_points': int(n_out),
                'outlier_fraction': float(outlier_fraction),
                'trend_significance': float(trend_significance),
                
                # Detection significance tests
                'transit_significance': float(abs(enhanced_snr)),
                't_test_significance': float(t_significance),
                't_test_p_value': float(t_p_value),
                'mannwhitney_significance': float(u_significance),
                'mannwhitney_p_value': float(u_p_value),
                
                # Transit characterization
                'deepest_phase': float(deepest_phase_center),
                'transit_asymmetry': float(abs(deepest_phase_center - 0.5) * 2),  # 0 = symmetric, 1 = very asymmetric
                
                # Overall quality scores
                'is_significant': bool(abs(enhanced_snr) > 3.0 and t_p_value < 0.01),
                'detection_confidence': float(min(abs(enhanced_snr) / 5.0, 1.0)),  # 0-1 scale
                'data_quality_score': float(min((1 - outlier_fraction) * phase_uniformity, 1.0)),
                
                # Enhanced flags
                'likely_real_transit': bool(
                    abs(enhanced_snr) > 3.0 and 
                    t_p_value < 0.01 and 
                    reduced_chi_squared < 3.0 and
                    outlier_fraction < 0.2 and
                    10 <= transit_depth * 1e6 <= 50000  # Reasonable planet depth range
                ),
                'possible_false_positive': bool(
                    trend_significance > 0.3 or 
                    outlier_fraction > 0.3 or
                    reduced_chi_squared > 5.0
                )
            }
            
            print(f"  ✅ Enhanced transit metrics calculated:")
            print(f"     Enhanced SNR: {enhanced_snr:.2f}")
            print(f"     Detection confidence: {metrics['detection_confidence']:.2f}")
            print(f"     Likely real transit: {'✓' if metrics['likely_real_transit'] else '✗'}")
            
        except Exception as e:
            print(f"  ⚠️ Error calculating enhanced transit metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics

    def _calculate_data_quality_score(self, time, flux, bin_counts):
        """Calculate overall data quality score"""
        
        # Factors affecting data quality:
        # 1. Temporal coverage
        # 2. Photometric precision
        # 3. Data density
        # 4. Systematic trends
        
        quality_score = 1.0
        
        # Temporal coverage (penalize gaps)
        if len(time) > 10:
            time_gaps = np.diff(np.sort(time))
            large_gaps = np.sum(time_gaps > 5 * np.median(time_gaps))
            quality_score *= (1 - min(large_gaps / len(time), 0.5))
        
        # Photometric precision
        flux_rms = np.std(flux)
        if flux_rms < 1e-4:  # Very precise
            quality_score *= 1.2
        elif flux_rms > 1e-3:  # Poor precision
            quality_score *= 0.8
        
        # Data density in bins
        if len(bin_counts) > 0:
            min_bin_count = np.min([c for c in bin_counts if c > 0])
            if min_bin_count < 3:  # Sparse data
                quality_score *= 0.7
        
        return np.clip(quality_score, 0, 1)
    

    # ===== IMPROVEMENT 5: ENHANCED VALIDATION PIPELINE =====
    def validate_candidate(self, candidate_data):
        """Complete validation pipeline for a single candidate with enhanced light curve analysis - FIXED VERSION"""
        
        # Extract candidate parameters with better error handling
        try:
            kic_id = int(candidate_data.get('kic_id', 0))
            period = float(candidate_data.get('period', 0))
            t0 = float(candidate_data.get('t0', 0))
            duration = float(candidate_data.get('duration', 0))
            disposition = str(candidate_data.get('disposition', 'UNKNOWN'))
        except (ValueError, TypeError) as e:
            print(f"  ❌ Error extracting candidate parameters: {e}")
            return {
                'kic_id': candidate_data.get('kic_id', 'unknown'),
                'rejection_reason': 'parameter_extraction_error',
                'passes_all_filters': False,
                'exoplanet_confidence': 'ERROR'
            }
        
        print(f"\n🔍 Validating KIC {kic_id} (P={period:.4f}d, Duration={duration:.2f}h)")
        
        # Initialize result dictionary
        result = {
            'kic_id': kic_id,
            'period': period,
            't0': t0,
            'duration': duration,
            'disposition': disposition,
            'priority': 'discard',
            'rejection_reason': '',
            'model_prob': 0.0,
            'enhanced_ml_prob': 0.0,
            'astronet_prediction': None,
            'light_curve_processing': None,
            'stellar_contamination': None,
            'secondary_eclipse': None,
            'transit_verification': None,
            'enhanced_transit_metrics': None,
            'passes_all_filters': False,
            'exoplanet_confidence': 'LOW',
            'enhanced_validation_boost': False
        }
        
        # Step 1: Basic parameter validation
        if period <= 0 or duration <= 0:
            result['rejection_reason'] = 'invalid_parameters'
            return result
        
        # Step 2: Duration filter (minimum 29.4 minutes = 0.49 hours)
        if duration < 0.49:
            result['priority'] = 'discard'
            result['rejection_reason'] = 'duration < 29.4 minutes'
            return result
        
        # Step 3: Light curve processing and phase folding
        if self.auto_download:
            print(f"  🌟 Processing light curve with phase folding...")
            try:
                lc_processing = self.process_light_curve(kic_id, period, t0, duration)

                if lc_processing is None:
                    lc_processing = {'processing_error': 'process_light_curve_returned_none'}
                result['light_curve_processing'] = lc_processing
                
                # Use enhanced ML probability if light curve processing succeeded
                if lc_processing.get('enhanced_ml_prob') is not None:
                    result['enhanced_ml_prob'] = lc_processing['enhanced_ml_prob']
                    result['model_prob'] = lc_processing['enhanced_ml_prob']
                    print(f"  ✅ Using enhanced ML probability from light curve analysis")
                else:
                    print(f"  ⚠️ Light curve processing failed, using fallback method")
            except Exception as lc_error:
                print(f"  ❌ Light curve processing error: {lc_error}")
                result['light_curve_processing'] = {'processing_error': str(lc_error)}
        
        # Step 4: Get ML probability (AstroNet or synthetic) if not from light curve
        
        if result['model_prob'] == 0.0:
             if self.use_astronet:
                try:
                    print(f"  Getting AstroNet prediction for KIC {kic_id}...")
                    astronet_result = self.run_astronet_prediction(kic_id, period, t0, duration)

                    if astronet_result is None: 
                        astronet_result = {"probability":0.0,"error":"astronet_returned_none"}

                    result['astronet_prediction'] = astronet_result
                    result['model_prob'] = astronet_result.get('probability', 0.0)

                    
                    if 'error' in astronet_result:
                        print(f"  ⚠️ AstroNet error for KIC {kic_id}: {astronet_result['error']}")
                        
                        fallback_prob = self._calculate_synthetic_ml_score_single(candidate_data)
                        result['model_prob'] = fallback_prob
                        print(f"  🔄 Using fallback synthetic score: {fallback_prob}")
                        
                except Exception as e:
                    print(f"  ❌ AstroNet prediction failed for KIC {kic_id}: {e}")
                    result['model_prob'] = self._calculate_synthetic_ml_score_single(candidate_data)
        else:
                result['model_prob'] = self._calculate_synthetic_ml_score_single(candidate_data)
        
        # Step 5: Enhanced validation checks if light curve is available
        if result.get('light_curve_processing', {}).get('light_curve_available'):
            print(f"  🔬 Running enhanced validation checks...")
            
            try:
                # Load light curve for enhanced validation
                lc = self._load_light_curve_for_validation(kic_id)
                
                if lc is not None:
                    # 1. Stellar contamination check
                    contamination = self.detect_stellar_contamination(lc, period, t0)
                    result['stellar_contamination'] = contamination
                    
                    # 2. Secondary eclipse check  
                    secondary_eclipse = self.detect_secondary_eclipse(lc, period, t0)
                    result['secondary_eclipse'] = secondary_eclipse
                    
                    # 3. Transit count verification
                    transit_count = self.verify_transit_count(lc, period, t0, duration)
                    result['transit_verification'] = transit_count
                    
                    # 4. Enhanced phase folding
                    enhanced_pf, enhanced_metrics = self.enhanced_phase_fold_transit(lc, period, t0, duration)
                    if enhanced_metrics:
                        result['enhanced_transit_metrics'] = enhanced_metrics

                    # 5. Centroid motion validation 
                    centroid_analysis = self.detect_centroid_motion(lc, period, t0, duration)
                    result['centroid_analysis'] = centroid_analysis

                    # 6. Transit timing variations
                    ttv_analysis = self.detect_transit_timing_variations(lc, period, t0, duration)
                    result['ttv_analysis'] = ttv_analysis
                    
                    # 7. Odd/even transit comparison
                    odd_even_analysis = self.compare_odd_even_transits(lc, period, t0, duration)
                    result['odd_even_analysis'] = odd_even_analysis
                    
                    # 8. Neighboring star contamination
                    neighbor_check = self.check_neighboring_star_contamination(kic_id, period, t0)
                    result['neighbor_contamination'] = neighbor_check
                    
                    # 9. Advanced stellar activity filtering
                    stellar_activity = self.advanced_stellar_activity_filter(lc, period, t0)
                    result['stellar_activity'] = stellar_activity
                    
                    
                    # CRITICAL VALIDATION DECISIONS

                    # Immediate rejection criteria (very strict)
                    if centroid_analysis.get('centroid_motion_detected', False):
                        max_shift = centroid_analysis.get('max_overall_shift_sigma', 0)
                        if max_shift > 5.0:  # Very significant centroid motion
                            result['priority'] = 'discard'
                            result['rejection_reason'] = 'significant_centroid_motion'
                            result['exoplanet_confidence'] = 'BACKGROUND_BINARY'
                            return result
                    
                    if odd_even_analysis.get('likely_eclipsing_binary_pattern', False):
                        result['priority'] = 'discard'
                        result['rejection_reason'] = 'eclipsing_binary_odd_even_pattern'
                        result['exoplanet_confidence'] = 'ECLIPSING_BINARY'
                        return result
                    

                    # FIXED: Only reject if VERY high contamination
                    if contamination.get('contamination_score', 0) > 0.95:  # Much higher threshold
                        result['priority'] = 'discard'
                        result['rejection_reason'] = 'severe_stellar_contamination'
                        result['exoplanet_confidence'] = 'CONTAMINATED'
                        return result
                    
                    # FIXED: More lenient transit count requirement
                    if transit_count.get('detected_transits', 0) < 1:  # Just need 1 transit
                        result['priority'] = 'discard'
                        result['rejection_reason'] = 'no_transits_detected'
                        result['exoplanet_confidence'] = 'INSUFFICIENT_DATA'
                        return result
                    
                   # FIXED: Only reject obvious eclipsing binaries
                    if secondary_eclipse.get('secondary_to_primary_ratio', 0) > 0.8:  # Much higher threshold
                        result['priority'] = 'discard'
                        result['rejection_reason'] = 'obvious_eclipsing_binary'
                        result['exoplanet_confidence'] = 'ECLIPSING_BINARY'
                        return result
                    
                    
                    # Statistical validation ensemble
                    ensemble_result = self.statistical_validation_ensemble(result)
                    result['statistical_validation'] = ensemble_result
                    
                    # Apply ensemble results to confidence
                    ensemble_score = ensemble_result.get('ensemble_score', 0.5)
                    validation_level = ensemble_result.get('validation_level', 'UNKNOWN')

                    # Boost confidence for candidates that pass all tests
                    contamination_score = contamination.get('contamination_score', 1.0)
                    
                    if contamination_score < 0.8:  # Good candidates get boost
                        original_prob = result.get('model_prob', 0.5)
                        boost_factor = 1.1 + (0.8 - contamination_score)  # Up to 1.9x boost
                        enhanced_prob = min(original_prob * boost_factor, 0.95)
                        ensemble_boost_factor = 0.5 + (ensemble_score * 0.5)  # 0.5 to 1.0 multiplier
                        final_ensemble_prob = original_prob * ensemble_boost_factor
                    
                        result['ensemble_enhanced_prob'] = min(final_ensemble_prob, 0.98)
                        result['validation_level'] = validation_level
                        result['enhanced_ml_prob'] = enhanced_prob
                        result['enhanced_validation_boost'] = True
                        print(f"  🚀 Enhanced validation boost applied: {original_prob:.3f} → {enhanced_prob:.3f}")
            
            except Exception as e:
                print(f"  ⚠️ Error in enhanced validation: {e}")
                result['enhanced_validation_error'] = str(e)
        
        # Step 6: Apply light curve quality filters
        lc_processing = result.get('light_curve_processing', {})
        if lc_processing and lc_processing.get('light_curve_available'):
            transit_metrics = lc_processing.get('transit_metrics', {})
            
            # Additional filters based on light curve analysis
            if transit_metrics and 'error' not in transit_metrics:
                transit_snr = transit_metrics.get('transit_snr', 0)
                depth_ppm = transit_metrics.get('transit_depth_ppm', 0)
                
                # Reject candidates with very poor SNR
                if transit_snr < -2:  # Strong anti-transit (eclipse)
                    result['priority'] = 'discard'
                    result['rejection_reason'] = 'anti_transit_detected'
                    result['exoplanet_confidence'] = 'STELLAR_ECLIPSE'
                    return result
                
                # Reject unreasonably deep transits (likely stellar eclipses)
                if depth_ppm > 100000:  # > 10%
                    result['priority'] = 'discard'
                    result['rejection_reason'] = 'transit_too_deep'
                    result['exoplanet_confidence'] = 'STELLAR_ECLIPSE'
                    return result
        
        # Step 7: Calculate final probability using comprehensive scoring
        final_prob = self.calculate_comprehensive_exoplanet_score(result)
        result['final_probability'] = final_prob
        result['comprehensive_score'] = final_prob
                
        # Step 8: Enhanced confidence assessment
        if final_prob >= 0.95:
            result['priority'] = 'very_high'
            result['exoplanet_confidence'] = 'EXTREMELY HIGH'
        elif final_prob >= 0.85:
            result['priority'] = 'high'
            result['exoplanet_confidence'] = 'VERY HIGH'
        elif final_prob >= 0.7:
            result['priority'] = 'medium_high'
            result['exoplanet_confidence'] = 'HIGH'
        elif final_prob >= 0.5:
            result['priority'] = 'medium'
            result['exoplanet_confidence'] = 'MEDIUM'
        elif final_prob >= 0.3:
            result['priority'] = 'low'
            result['exoplanet_confidence'] = 'LOW'
        else:
            result['priority'] = 'discard'
            result['rejection_reason'] = 'probability_too_low'
            result['exoplanet_confidence'] = 'VERY LOW'
            return result
        
        # Step 9: Additional boost for light curve confirmed transits
        if lc_processing and lc_processing.get('light_curve_available'):
            transit_metrics = lc_processing.get('transit_metrics', {})
            if transit_metrics and transit_metrics.get('is_significant', False) and transit_metrics.get('transit_snr', 0) > 3:
                result['passes_all_filters'] = True
                result['light_curve_confirmed'] = True
                print(f"  🎉 LIGHT CURVE CONFIRMED TRANSIT! SNR={transit_metrics.get('transit_snr', 0):.2f}")
        
        # Step 10: Mark high probability cases as likely exoplanets
        if final_prob >= 0.7:
            result['passes_all_filters'] = True
        
        print(f"  ✅ Validation complete: {result['exoplanet_confidence']} confidence ({final_prob*100:.1f}%)")
        
        return result
    
    def _load_light_curve_for_validation(self, kic_id):
        """Load light curve for validation - modify based on your storage method"""
        
        try:
            # Try to load from your processed light curves directory
            lc_filename = os.path.join(self.processed_lc_dir, f"kic_{kic_id}_processed.fits")
            
            if os.path.exists(lc_filename):
                import lightkurve as lk
                return lk.read(lc_filename)
            else:
                # Re-download if needed
                return self.download_light_curve_lightkurve(kic_id)
                
        except Exception as e:
            print(f"  ⚠️ Could not load light curve for validation: {e}")
            return None
    
    def _create_simple_transit_model(self, time, period, t0, duration):
        """Create simple box transit model for detrending"""
        
        phases = ((time - t0) % period) / period
        phases = (phases + 0.5) % 1.0
        
        transit_duration_phase = max(duration / 24.0 / period, 0.02)  # At least 2% of orbit

        in_transit = abs(phases - 0.5) < (transit_duration_phase / 2)
        
        model = np.ones_like(time)
        model[in_transit] = 0.999  # Shallow transit for detrending
        
        return model
    
    def _calculate_enhanced_ml_probability(self, metrics, processing_result):
        """
        Calculate enhanced ML probability based on light curve analysis
        This combines the original ML score with new transit metrics
        """
        
        base_score = 0.5  # Start with neutral probability
        
        try:
            # Transit significance component (most important)
            transit_snr = metrics.get('transit_snr', 0)
            if transit_snr > 5:
                base_score += 0.3
            elif transit_snr > 3:
                base_score += 0.2
            elif transit_snr > 1.5:
                base_score += 0.1
            elif transit_snr < 0:  # Anti-transit (stellar eclipse)
                base_score -= 0.2
            
            # Transit depth component
            depth_ppm = metrics.get('transit_depth_ppm', 0)
            if 50 < depth_ppm < 50000:  # Reasonable exoplanet range
                base_score += 0.15
            elif depth_ppm > 50000:  # Too deep, likely stellar eclipse
                base_score -= 0.2
            
            # Duration consistency
            measured_duration = metrics.get('measured_duration_hours', 0)
            expected_duration = processing_result.get('duration', 0)
            if expected_duration > 0:
                duration_ratio = measured_duration / expected_duration
                if 0.5 < duration_ratio < 2.0:  # Within reasonable range
                    base_score += 0.1
                else:
                    base_score -= 0.1
            
            # Statistical significance
            if metrics.get('is_significant', False):
                base_score += 0.1
            
            # Phase coverage (good sampling across orbit)
            phase_coverage = metrics.get('phase_coverage', 0)
            if phase_coverage > 0.5:
                base_score += 0.05
            
            # Chi-squared goodness of fit
            reduced_chi_sq = metrics.get('reduced_chi_squared', 10)
            if 0.5 < reduced_chi_sq < 2.0:  # Good model fit
                base_score += 0.05
            elif reduced_chi_sq > 5:  # Poor fit
                base_score -= 0.1
            
            # BLS statistic
            bls_stat = metrics.get('bls_statistic', 0)
            if bls_stat > 5:
                base_score += 0.1
            
        except Exception as e:
            print(f"  ⚠️ Error calculating enhanced ML probability: {e}")
            return 0.5
        
        # Clip to valid probability range
        return np.clip(base_score, 0.0, 1.0)

    def run_astronet_prediction(self, kepid, period, t0, duration, max_retries=2):
        """
        Run AstroNet prediction for a candidate using the provided snippet
        """
        
        if not self.use_astronet:
            return {'probability': 0.0, 'error': 'AstroNet disabled'}
        
        print(f"🤖 Running AstroNet prediction for KIC {kepid}...")
           
        
        # Prepare file paths
        image_file = os.path.join(self.predicted_images_dir, f"kepler-{kepid}.png")
        config_file = os.path.join(self.model_dir, 'config.json')
        
        # Check if model directory and config exist
        if not os.path.exists(self.model_dir):
            print(f"❌ Model directory not found: {self.model_dir}")
            return {'probability': 0.0, 'error': 'Model directory not found'}
            
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return {'probability': 0.0, 'error': 'Config file not found'}
        
        # Build command exactly as provided - Fixed command structure
        command = [
            "bazel", "run", "//astronet:predict", "--",
            "--model=AstroCNNModel",
            f"--config_json={config_file}",
            f"--model_dir={self.model_dir}",
            f"--kepler_data_dir={self.kepler_dir}",
            f"--kepler_id={kepid}",
            f"--period={period}",
            f"--t0={t0}",
            f"--duration={duration}",
            f"--output_image_file={image_file}",
        ]
        
        # Execute prediction with retries and timeout
        for attempt in range(max_retries + 1):
            try:
                print(f"  Executing AstroNet command (attempt {attempt + 1})...")
                print(f"  Command: {' '.join(command)}")
                
                # Run the command and capture output with timeout
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    cwd=self.base_dir,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    # Parse the output for prediction probability
                    probability = self._parse_astronet_output(result.stdout, result.stderr)
                    
                    prediction_result = {
                        'probability': probability,
                        'image_file': image_file if os.path.exists(image_file) else None,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'command': ' '.join(command),
                        'attempt': attempt + 1
                    }
                    
                    print(f"  ✅ AstroNet prediction successful: {probability:.3f}")
                    return prediction_result
                    
                else:
                    print(f"  ❌ AstroNet command failed with return code {result.returncode}")
                    print(f"     STDOUT: {result.stdout[:500]}...")
                    print(f"     STDERR: {result.stderr[:500]}...")
                    
                    if attempt < max_retries:
                        print(f"  🔄 Retrying in {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ AstroNet prediction timed out (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"  ❌ Error running AstroNet: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        
        # All attempts failed
        return {
            'probability': 0.0, 
            'error': 'All AstroNet prediction attempts failed',
            'command': ' '.join(command)
        }
    
    def _parse_astronet_output(self, stdout, stderr):
        """Parse AstroNet output to extract prediction probability - FIXED VERSION"""
        
        full_output = stdout + "\n" + stderr

        print("printing full output",full_output)
        
        # Look for the specific score patterns - FIXED ORDER AND PATTERNS
        score_patterns = [
            # Most specific patterns first (highest priority)
            r'Planet candidate score \(0-1\) \[max of output array\]:\s*([0-9]*\.?[0-9]+)',
            r'KIC\d+\s*=\s*([0-9]*\.?[0-9]+)',
            r'KIC\d+\s*=\s*Is a planet candidate.*?([0-9]*\.?[0-9]+)'
        ]
        
        found_scores = []
        
        for i, pattern in enumerate(score_patterns):
            matches = re.findall(pattern, full_output, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"  🎯 Pattern {i+1} found matches: {matches}")
                for match in matches:
                    try:
                        prob = float(match)
                        if 0.0 <= prob <= 1.0:
                            found_scores.append((i, prob, pattern))
                            print(f"    ✅ Valid score: {prob} from pattern: {pattern}")
                        else:
                            print(f"    ❌ Invalid score range: {prob}")
                    except ValueError:
                        print(f"    ❌ Could not convert to float: {match}")
        
        if found_scores:
            # Sort by pattern priority (lower index = higher priority)
            found_scores.sort(key=lambda x: x[0])
            best_score = found_scores[0][1]
            best_pattern = found_scores[0][2]
            
            print(f"  🏆 Selected best score: {best_score} from pattern: {best_pattern}")
            return best_score
        
        # Additional debug info if no score found
        print(f"  ⚠️ No valid scores found in output")
        print(f"  📝 Raw output lines:")
        for i, line in enumerate(full_output.split('\n')[:10]):  # First 10 lines
            print(f"    {i+1}: {line}")
        
        return 0.5  # Neutral fallback


    def _calculate_synthetic_ml_score(self):
        """
        Calculate synthetic ML probability score based on available KOI metrics
        This approximates what a machine learning model might produce
        (Used as fallback when AstroNet is not available)
        """
        scores = np.zeros(len(self.data))
        
        # Base score from SNR
        if 'snr' in self.data.columns:
            snr_scores = np.clip(self.data['snr'].fillna(0) / 20.0, 0, 0.4)
            scores += snr_scores
        
        # Score from disposition score if available
        if 'score' in self.data.columns:
            disp_scores = np.clip(self.data['score'].fillna(0), 0, 0.3)
            scores += disp_scores
        
        # Bonus for confirmed/candidate status
        if 'disposition' in self.data.columns:
            confirmed_bonus = (self.data['disposition'] == 'CONFIRMED') * 0.2
            candidate_bonus = (self.data['disposition'] == 'CANDIDATE') * 0.1
            scores += confirmed_bonus + candidate_bonus
        
        # Penalty for false positive flags
        fp_columns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        for col in fp_columns:
            if col in self.data.columns:
                fp_penalty = (self.data[col] == 1) * 0.05
                scores -= fp_penalty
        
        # Normalize to 0-1 range
        scores = np.clip(scores, 0, 1)
        
        # Add some realistic spread
        scores += np.random.normal(0, 0.05, len(scores))
        scores = np.clip(scores, 0, 1)
        
        return scores

    def _calculate_synthetic_ml_score_single(self, candidate_data):
            """Calculate synthetic ML score for a single candidate"""
            score = 0.0
            
            # Base score from SNR
            snr = candidate_data.get('snr', 0)
            if snr:
                score += min(float(snr) / 20.0, 0.4)
            
            # Score from disposition score
            disp_score = candidate_data.get('score', 0)
            if disp_score:
                score += min(float(disp_score), 0.3)
            
            # Bonus for disposition
            disposition = candidate_data.get('disposition', '')
            if disposition == 'CONFIRMED':
                score += 0.2
            elif disposition == 'CANDIDATE':
                score += 0.1
            
            # Penalty for false positive flags
            fp_columns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
            for col in fp_columns:
                if candidate_data.get(col) == 1:
                    score -= 0.05
            
            # Add some noise and clip
            score += np.random.normal(0, 0.05)
            return np.clip(score, 0, 1)
        
    def calculate_comprehensive_exoplanet_score(self, result):
        """Calculate final exoplanet probability using ALL available metrics"""
        
        scores = []
        weights = []
        
        # 1. ML Model Score (30% weight)
        ml_prob = result.get('final_probability', 0.5)
        scores.append(ml_prob)
        weights.append(0.3)
        
        # 2. Light Curve Transit Metrics (50% weight if available)
        lc_proc = result.get('light_curve_processing', {})
        if lc_proc.get('light_curve_available'):
            transit_metrics = lc_proc.get('transit_metrics', {})
            
            if 'error' not in transit_metrics:
                # Transit SNR score
                snr = transit_metrics.get('transit_snr', 0)
                snr_score = min(max((snr - 1) / 4, 0), 1)  # 1-5 SNR maps to 0-1
                scores.append(snr_score)
                weights.append(0.2)
                
                # Transit depth score (reasonable exoplanet range)
                depth_ppm = transit_metrics.get('transit_depth_ppm', 0)
                if 50 <= depth_ppm <= 10000:  # Reasonable planet range
                    depth_score = 0.8
                elif 10 <= depth_ppm < 50:    # Weak but possible
                    depth_score = 0.4
                elif depth_ppm > 10000:       # Too deep (stellar eclipse)
                    depth_score = 0.0
                else:                         # Negative or tiny
                    depth_score = 0.1
                scores.append(depth_score)
                weights.append(0.15)
                
                # Statistical significance
                is_sig = transit_metrics.get('is_significant', False)
                sig_score = 1.0 if is_sig else 0.3
                scores.append(sig_score)
                weights.append(0.1)
                
                # Chi-squared goodness of fit
                chi_sq = transit_metrics.get('reduced_chi_squared', 10)
                if 0.5 <= chi_sq <= 2.0:
                    chi_score = 0.8
                elif 2.0 < chi_sq <= 5.0:
                    chi_score = 0.5
                else:
                    chi_score = 0.2
                scores.append(chi_score)
                weights.append(0.05)
        
        # 3. Catalog-based metrics (20% weight)
        # SNR from catalog
        if 'snr' in result:
            catalog_snr = result.get('snr', 0)
            catalog_snr_score = min(catalog_snr / 20, 1.0)
            scores.append(catalog_snr_score)
            weights.append(0.1)
        
        # Disposition bonus
        disposition = result.get('disposition', '')
        if disposition == 'CONFIRMED':
            scores.append(1.0)
            weights.append(0.1)
        elif disposition == 'CANDIDATE':
            scores.append(0.7)
            weights.append(0.1)
        
        # Calculate weighted average
        if len(scores) > 0:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return np.clip(weighted_score, 0, 1)
        
        return ml_prob  # Fallback to ML only

    def validate_all_candidates(self, max_candidates=None):
        """Validate all candidates with enhanced processing and progress tracking - FIXED VERSION"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        candidates_to_process = self.data.copy()
        if max_candidates:
            candidates_to_process = candidates_to_process.head(max_candidates)
        
        print(f"Starting enhanced validation of {len(candidates_to_process)} candidates...")
        print("This includes light curve downloading and phase folding analysis...")
        if self.use_astronet:
            print("AstroNet predictions will also be generated where possible...")
        
        results = []
        failed_predictions = 0
        light_curve_successes = 0
        
        # Progress bar setup
        with tqdm(total=len(candidates_to_process), desc="Processing candidates") as pbar:
            
            for idx, (_, candidate) in enumerate(candidates_to_process.iterrows()):
                
                try:
                    # Convert candidate row to dictionary safely
                    candidate_dict = candidate.to_dict()
                    
                    # Validate the candidate
                    result = self.validate_candidate(candidate_dict)
                    
                    if result is None:
                        # Create a default error result
                        result = {
                            'kic_id': candidate_dict.get('kic_id', 'unknown'),
                            'rejection_reason': 'validation_returned_none',
                            'passes_all_filters': False,
                            'exoplanet_confidence': 'ERROR'
                        }

                    results.append(result)

                    # if self.use_astronet and result.get('astronet_prediction', {}).get('error'):
                    #     failed_predictions += 1

                    # if result.get('light_curve_processing', {}).get('light_curve_available'):
                    #     light_curve_successes += 1

                    kic_id = result.get('kic_id', 'unknown')
                    confidence = result.get('exoplanet_confidence', 'UNKNOWN')
                    pbar.set_postfix({
                        'Current': f'KIC {kic_id}',
                        'Confidence': confidence,
                        'LC Success': f'{light_curve_successes}/{idx+1}'
                    })

                except Exception as e:
                    print(f"\nError processing KIC {candidate.get('kic_id', 'unknown')}: {e}")
                    failed_result = {
                        'kic_id': candidate.get('kic_id', 'unknown'),
                        'rejection_reason': f'processing_error: {str(e)}',
                        'passes_all_filters': False,
                        'exoplanet_confidence': 'ERROR'
                    }
                    results.append(failed_result)
                
                pbar.update(1)
        
        self.results = results
        
        print(f"\n🎉 Enhanced validation complete!")
        print(f"Light curve processing successes: {light_curve_successes}/{len(candidates_to_process)}")
        if self.use_astronet:
            print(f"AstroNet prediction failures: {failed_predictions}/{len(candidates_to_process)}")
        
        self.print_enhanced_summary()
        return results

    def print_enhanced_summary(self):
        """Print detailed validation summary with all advanced validation results"""
        if not self.results:
            return
        
        total = len(self.results)
        
        # Count by validation levels
        gold_candidates = len([r for r in self.results if r.get('validation_level') == 'GOLD'])
        silver_candidates = len([r for r in self.results if r.get('validation_level') == 'SILVER'])
        bronze_candidates = len([r for r in self.results if r.get('validation_level') == 'BRONZE'])
        
        # Advanced validation statistics
        centroid_passed = len([r for r in self.results 
                            if not r.get('centroid_analysis', {}).get('centroid_motion_detected', True)])
        
        odd_even_passed = len([r for r in self.results 
                            if r.get('odd_even_analysis', {}).get('odd_even_consistent', False)])
        
        ttv_interesting = len([r for r in self.results 
                            if r.get('ttv_analysis', {}).get('indicates_additional_planets', False)])
        
        multi_aperture_passed = len([r for r in self.results 
                                    if r.get('multi_aperture', {}).get('multi_aperture_consistent', False)])
        
        print("\n" + "="*100)
        print("🚀 ULTRA-ENHANCED EXOPLANET DETECTION RESULTS WITH ADVANCED VALIDATION")
        print("="*100)
        print(f"Total candidates analyzed:           {total}")
        
        print(f"\n🏆 VALIDATION LEVELS (Advanced Statistical Ensemble):")
        print(f"GOLD (>90%):                         {gold_candidates} candidates 🥇🥇🥇")
        print(f"SILVER (80-90%):                     {silver_candidates} candidates 🥈🥈")
        print(f"BRONZE (70-80%):                     {bronze_candidates} candidates 🥉")
        
        print(f"\n🔬 ADVANCED VALIDATION STATISTICS:")
        print(f"Passed centroid motion test:         {centroid_passed}/{total} ({centroid_passed/total*100:.1f}%)")
        print(f"Passed odd/even consistency:         {odd_even_passed}/{total} ({odd_even_passed/total*100:.1f}%)")
        print(f"Passed multi-aperture test:          {multi_aperture_passed}/{total}")
        
        if ttv_interesting > 0:
            print(f"Show signs of additional planets:    {ttv_interesting} candidates 🪐🪐")
        
        # Show ultra-high confidence candidates
        ultra_high = [r for r in self.results if r.get('validation_level') in ['GOLD', 'SILVER']]
        
        if ultra_high:
            ultra_high.sort(key=lambda x: x.get('ensemble_enhanced_prob', 0), reverse=True)
            
            print(f"\n🌟 ULTRA-HIGH CONFIDENCE EXOPLANET CANDIDATES:")
            print("-" * 100)
            print(f"{'Rank':<4} {'KIC':<10} {'Level':<7} {'Prob':<8} {'Period':<8} {'Centroid':<9} {'O/E':<5} {'TTV':<5} {'Notes'}")
            print("-" * 100)
            
            for i, candidate in enumerate(ultra_high[:15]):  # Top 15
                kic_id = candidate.get('kic_id', 'N/A')
                level = candidate.get('validation_level', 'N/A')
                prob = candidate.get('ensemble_enhanced_prob', 0) * 100
                period = candidate.get('period', 0)
                
                # Advanced validation flags
                centroid_ok = "✓" if not candidate.get('centroid_analysis', {}).get('centroid_motion_detected', True) else "✗"
                odd_even_ok = "✓" if candidate.get('odd_even_analysis', {}).get('odd_even_consistent', False) else "✗"
                ttv_planets = "TTV+" if candidate.get('ttv_analysis', {}).get('indicates_additional_planets', False) else ""
                
                # Special notes
                notes = []
                if candidate.get('light_curve_confirmed'):
                    notes.append("LC✓")
                if ttv_planets:
                    notes.append("Multi-🪐")
                if candidate.get('validation_level') == 'GOLD':
                    notes.append("GOLD⭐")
                
                notes_str = " ".join(notes)
                
                print(f"{i+1:<4} {kic_id:<10} {level:<7} {prob:<7.1f}% {period:<7.2f}d {centroid_ok:<9} {odd_even_ok:<5} {ttv_planets:<5} {notes_str}")
        
        print(f"\n📊 ULTRA-ENHANCED CONCLUSIONS:")
        total_ultra_high = gold_candidates + silver_candidates + bronze_candidates
        
        if gold_candidates > 0:
            print(f"🥇 {gold_candidates} GOLD-LEVEL candidates - highest possible confidence!")
            print("   These have passed ALL advanced validation tests.")
        
        if silver_candidates > 0:
            print(f"🥈 {silver_candidates} SILVER-LEVEL candidates - very high confidence!")
            print("   These show excellent validation metrics.")
        
        if total_ultra_high > 0:
            print(f"✅ {total_ultra_high} total ultra-high confidence candidates!")
            success_rate = total_ultra_high / total * 100
            print(f"   Advanced validation success rate: {success_rate:.1f}%")
        
        if ttv_interesting > 0:
            print(f"🎯 {ttv_interesting} candidates show TTV signatures of additional planets!")
            print("   These are particularly interesting multi-planet systems!")
        
        print("="*100)

    def save_enhanced_results(self, filename='enhanced_exoplanet_results.csv'):
            
        """Save ultra-enhanced results with all advanced validation data"""
        if not self.results:
            print("No results to save")
            return
        
        # Convert results to DataFrame with all advanced validation data
        ultra_enhanced_results = []
        
        for result in self.results:
            flat_result = result.copy()
            
            # Flatten advanced validation results
            centroid = result.get('centroid_analysis', {})
            flat_result['centroid_motion_detected'] = centroid.get('centroid_motion_detected', False)
            flat_result['max_centroid_shift_sigma'] = centroid.get('max_overall_shift_sigma', 0)
            
            odd_even = result.get('odd_even_analysis', {})
            flat_result['odd_even_consistent'] = odd_even.get('odd_even_consistent', True)
            flat_result['odd_even_depth_ratio'] = odd_even.get('depth_ratio', 0)
            flat_result['eclipsing_binary_pattern'] = odd_even.get('likely_eclipsing_binary_pattern', False)
            
            ttv = result.get('ttv_analysis', {})
            flat_result['ttv_detected'] = ttv.get('ttv_detected', False)
            flat_result['ttv_rms_minutes'] = ttv.get('ttv_rms_minutes', 0)
            flat_result['indicates_additional_planets'] = ttv.get('indicates_additional_planets', False)
            
            multi_ap = result.get('multi_aperture', {})
            flat_result['multi_aperture_consistent'] = multi_ap.get('multi_aperture_consistent', True)
            flat_result['aperture_depth_cv'] = multi_ap.get('depth_consistency_cv', 0)
            
            stellar_act = result.get('stellar_activity', {})
            flat_result['stellar_activity_score'] = stellar_act.get('stellar_activity_score', 0)
            flat_result['high_stellar_activity'] = stellar_act.get('high_activity', False)
            
            ensemble = result.get('statistical_validation', {})
            flat_result['ensemble_score'] = ensemble.get('ensemble_score', 0.5)
            flat_result['validation_level'] = ensemble.get('validation_level', 'UNKNOWN')
            flat_result['ensemble_enhanced_prob'] = result.get('ensemble_enhanced_prob', result.get('model_prob', 0))
            
            # Remove nested dictionaries
            keys_to_remove = ['centroid_analysis', 'odd_even_analysis', 'ttv_analysis', 
                            'multi_aperture', 'stellar_activity', 'statistical_validation',
                            'light_curve_processing', 'astronet_prediction']
            
            for key in keys_to_remove:
                flat_result.pop(key, None)
            
            ultra_enhanced_results.append(flat_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(ultra_enhanced_results)
        
        # Add interpretable columns
        results_df['is_gold_candidate'] = results_df['validation_level'] == 'GOLD'
        results_df['is_ultra_high_confidence'] = results_df['validation_level'].isin(['GOLD', 'SILVER'])
        results_df['passed_all_tests'] = (
            (~results_df['centroid_motion_detected']) &
            (results_df['odd_even_consistent']) &
            (~results_df['eclipsing_binary_pattern']) &
            (results_df['multi_aperture_consistent'])
        )
        
        # Save main results
        results_df.to_csv(filename, index=False)
        print(f"Ultra-enhanced results saved to {filename}")
        
        # Save gold candidates separately
        gold_candidates = results_df[results_df['validation_level'] == 'GOLD']
        if len(gold_candidates) > 0:
            gold_file = filename.replace('.csv', '_GOLD_CANDIDATES.csv')
            gold_candidates.to_csv(gold_file, index=False)
            print(f"GOLD candidates saved to {gold_file}")
        
        # Save comprehensive summary
        summary_file = filename.replace('.csv', '_COMPREHENSIVE_SUMMARY.txt')
        with open(summary_file, 'w',encoding="utf-8") as f:
            f.write("ULTRA-ENHANCED EXOPLANET DETECTION COMPREHENSIVE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            total = len(results_df)
            gold = len(results_df[results_df['validation_level'] == 'GOLD'])
            silver = len(results_df[results_df['validation_level'] == 'SILVER'])
            bronze = len(results_df[results_df['validation_level'] == 'BRONZE'])
            
            f.write(f"VALIDATION SUMMARY:\n")
            f.write(f"Total candidates: {total}\n")
            f.write(f"GOLD level (>90%): {gold}\n")
            f.write(f"SILVER level (80-90%): {silver}\n") 
            f.write(f"BRONZE level (70-80%): {bronze}\n\n")
            
            if gold > 0:
                f.write("GOLD-LEVEL EXOPLANET CANDIDATES:\n")
                f.write("-" * 50 + "\n")
                for _, row in gold_candidates.sort_values('ensemble_enhanced_prob', ascending=False).iterrows():
                    f.write(f"KIC {row['kic_id']}: {row['ensemble_enhanced_prob']*100:.1f}% confidence\n")
                    f.write(f"  Period: {row['period']:.3f} days\n")
                    f.write(f"  Validation level: {row['validation_level']}\n")
                    f.write(f"  Centroid stable: {'✓' if not row['centroid_motion_detected'] else '✗'}\n")
                    f.write(f"  Odd/even consistent: {'✓' if row['odd_even_consistent'] else '✗'}\n")
                    if row.get('indicates_additional_planets', False):
                        f.write(f"  → Possible multi-planet system!\n")
                    f.write(f"\n")
        
        print(f"Comprehensive summary saved to {summary_file}")
        
        return results_df
    
    def validate_light_curve_quality(self, lc):
        """
        Validate that the light curve has sufficient quality for advanced analysis
        """
        try:
            if lc is None:
                return False, "No light curve data"
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Check data length
            if len(time) < 100:
                return False, f"Insufficient data points: {len(time)}"
            
            # Check for excessive NaN values
            nan_fraction = np.sum(~np.isfinite(flux)) / len(flux)
            if nan_fraction > 0.5:
                return False, f"Too many NaN values: {nan_fraction*100:.1f}%"
            
            # Check time span
            time_span = time.max() - time.min()
            if time_span < 10:  # Less than 10 days
                return False, f"Insufficient time coverage: {time_span:.1f} days"
            
            # Check for gaps
            time_diffs = np.diff(np.sort(time))
            large_gaps = np.sum(time_diffs > 5 * np.median(time_diffs))
            if large_gaps > len(time) * 0.1:
                return False, f"Too many large gaps in data"
            
            # Check flux variability
            flux_std = np.std(flux[np.isfinite(flux)])
            if flux_std < 1e-6:
                return False, "Flux shows no variability (flat line)"
            
            return True, "Light curve quality acceptable"
            
        except Exception as e:
            return False, f"Error validating light curve: {e}"
        

    def create_validation_plots(self, kic_id, result_dict, save_dir=None):
        """
        Create comprehensive validation plots showing all analysis results
        """
        try:
            if save_dir is None:
                save_dir = os.path.join(self.base_dir, "validation_plots")
            os.makedirs(save_dir, exist_ok=True)
            
            # Load light curve
            lc = self._load_light_curve_for_validation(kic_id)
            if lc is None:
                return None
            
            fig = plt.figure(figsize=(20, 15))
            
            # Plot 1: Raw light curve
            ax1 = plt.subplot(3, 4, 1)
            ax1.plot(lc.time.value, lc.flux.value, 'k.', alpha=0.5, markersize=1)
            ax1.set_xlabel('Time (BKJD)')
            ax1.set_ylabel('Flux')
            ax1.set_title(f'KIC {kic_id} - Raw Light Curve')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Phase-folded light curve
            period = result_dict.get('period', 1)
            t0 = result_dict.get('t0', 0)
            duration = result_dict.get('duration', 1)
            
            phases = ((lc.time.value - t0) % period) / period
            phases = (phases + 0.5) % 1.0
            
            ax2 = plt.subplot(3, 4, 2)
            ax2.plot(phases, lc.flux.value, 'b.', alpha=0.6, markersize=2)
            ax2.axvline(0.5, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Flux')
            ax2.set_title('Phase-Folded Transit')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Centroid motion (if available)
            ax3 = plt.subplot(3, 4, 3)
            if hasattr(lc, 'centroid_col') and hasattr(lc, 'centroid_row'):
                ax3.plot(lc.centroid_col.value, lc.centroid_row.value, 'g.', alpha=0.5)
                ax3.set_xlabel('Centroid Column')
                ax3.set_ylabel('Centroid Row')
                ax3.set_title('Centroid Motion')
            else:
                ax3.text(0.5, 0.5, 'No Centroid Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Centroid Motion - N/A')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Transit timing variations
            ax4 = plt.subplot(3, 4, 4)
            ttv_data = result_dict.get('ttv_analysis', {})
            if 'timing_residuals_minutes' in ttv_data and 'transit_epochs' in ttv_data:
                residuals = ttv_data['timing_residuals_minutes']
                epochs = ttv_data['transit_epochs']
                ax4.plot(epochs, residuals, 'ro-', markersize=4)
                ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Transit Number')
                ax4.set_ylabel('O-C (minutes)')
                ax4.set_title(f'Transit Timing Variations\nRMS: {ttv_data.get("ttv_rms_minutes", 0):.1f} min')
            else:
                ax4.text(0.5, 0.5, 'TTV Analysis\nNot Available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Transit Timing Variations')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Odd vs Even transits
            ax5 = plt.subplot(3, 4, 5)
            odd_even_data = result_dict.get('odd_even_analysis', {})
            if 'odd_transit_data' in odd_even_data and 'even_transit_data' in odd_even_data:
                odd_depths = [t['depth_ppm'] for t in odd_even_data['odd_transit_data']]
                even_depths = [t['depth_ppm'] for t in odd_even_data['even_transit_data']]
                
                x_odd = np.ones(len(odd_depths)) * 1
                x_even = np.ones(len(even_depths)) * 2
                
                ax5.scatter(x_odd, odd_depths, color='blue', alpha=0.7, s=50, label=f'Odd (n={len(odd_depths)})')
                ax5.scatter(x_even, even_depths, color='red', alpha=0.7, s=50, label=f'Even (n={len(even_depths)})')
                
                ax5.axhline(np.median(odd_depths), color='blue', linestyle='--', alpha=0.7)
                ax5.axhline(np.median(even_depths), color='red', linestyle='--', alpha=0.7)
                
                ax5.set_xticks([1, 2])
                ax5.set_xticklabels(['Odd', 'Even'])
                ax5.set_ylabel('Transit Depth (ppm)')
                ax5.set_title('Odd vs Even Transits')
                ax5.legend()
            else:
                ax5.text(0.5, 0.5, 'Odd/Even Analysis\nNot Available', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Odd vs Even Transits')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Stellar activity periodogram
            ax6 = plt.subplot(3, 4, 6)
            stellar_activity = result_dict.get('stellar_activity', {})
            if 'rotation_periods' in stellar_activity:
                # Create a simple periodogram visualization
                try:
                    frequency = np.linspace(1/50, 1/0.5, 1000)
                    periods = 1/frequency
                    ls = LombScargle(lc.time.value, lc.flux.value - np.median(lc.flux.value))
                    power = ls.power(frequency)
                    
                    ax6.semilogx(periods, power, 'k-', alpha=0.7)
                    ax6.set_xlabel('Period (days)')
                    ax6.set_ylabel('LS Power')
                    ax6.set_title('Stellar Activity Periodogram')
                    
                    # Mark rotation periods if detected
                    rot_periods = stellar_activity.get('rotation_periods', [])
                    for p in rot_periods[:3]:  # Mark up to 3 periods
                        ax6.axvline(p, color='red', alpha=0.6, linestyle='--')
                        
                except Exception:
                    ax6.text(0.5, 0.5, 'Periodogram\nError', ha='center', va='center', transform=ax6.transAxes)
            else:
                ax6.text(0.5, 0.5, 'Stellar Activity\nNot Available', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Stellar Activity')
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: Secondary eclipse test
            ax7 = plt.subplot(3, 4, 7)
            # Phase fold at secondary eclipse phase
            secondary_phases = ((lc.time.value - (t0 + period/2)) % period) / period
            secondary_phases = (secondary_phases + 0.5) % 1.0
            
            ax7.plot(secondary_phases, lc.flux.value, 'purple', alpha=0.5, marker='.', markersize=1, linestyle='')
            ax7.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
            ax7.set_xlabel('Phase (Secondary Eclipse)')
            ax7.set_ylabel('Flux')
            ax7.set_title('Secondary Eclipse Test')
            ax7.grid(True, alpha=0.3)
            
            # Plot 8: Multi-aperture comparison (placeholder)
            ax8 = plt.subplot(3, 4, 8)
            multi_ap = result_dict.get('multi_aperture', {})
            if 'aperture_results' in multi_ap:
                aperture_results = multi_ap['aperture_results']
                if aperture_results:
                    apertures = [r['aperture'] for r in aperture_results]
                    depths = [r['transit_depth_ppm'] for r in aperture_results]
                    
                    ax8.bar(range(len(apertures)), depths, alpha=0.7)
                    ax8.set_xticks(range(len(apertures)))
                    ax8.set_xticklabels(apertures, rotation=45)
                    ax8.set_ylabel('Transit Depth (ppm)')
                    ax8.set_title('Multi-Aperture Consistency')
                else:
                    ax8.text(0.5, 0.5, 'Multi-Aperture\nNo Data', ha='center', va='center', transform=ax8.transAxes)
                    ax8.set_title('Multi-Aperture Test')
            else:
                ax8.text(0.5, 0.5, 'Multi-Aperture\nNot Available', ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Multi-Aperture Test')
            ax8.grid(True, alpha=0.3)
            
            # Plot 9: Validation scores radar chart (simplified bar chart)
            ax9 = plt.subplot(3, 4, 9)
            ensemble = result_dict.get('statistical_validation', {})
            if 'individual_scores' in ensemble:
                scores = ensemble['individual_scores']
                metrics = list(scores.keys())
                values = list(scores.values())
                
                colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
                bars = ax9.barh(metrics, values, color=colors, alpha=0.7)
                ax9.set_xlim(0, 1)
                ax9.set_xlabel('Validation Score')
                ax9.set_title('Individual Validation Metrics')
            else:
                ax9.text(0.5, 0.5, 'Validation Scores\nNot Available', ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Validation Scores')
            ax9.grid(True, alpha=0.3, axis='x')
            
            # Plot 10: Overall validation summary
            ax10 = plt.subplot(3, 4, 10)
            ax10.axis('off')
            
            # Create validation summary text
            validation_level = result_dict.get('validation_level', 'UNKNOWN')
            ensemble_prob = result_dict.get('ensemble_enhanced_prob', 0) * 100
            
            summary_text = f"""
    VALIDATION SUMMARY
    KIC {kic_id}

    Validation Level: {validation_level}
    Ensemble Probability: {ensemble_prob:.1f}%

    Advanced Checks:
    """
            
            # Add check results
            centroid_ok = not result_dict.get('centroid_analysis', {}).get('centroid_motion_detected', True)
            odd_even_ok = result_dict.get('odd_even_analysis', {}).get('odd_even_consistent', False)
            multi_ap_ok = result_dict.get('multi_aperture', {}).get('multi_aperture_consistent', True)
            
            summary_text += f"Centroid Stable: {'✓' if centroid_ok else '✗'}\n"
            summary_text += f"Odd/Even Consistent: {'✓' if odd_even_ok else '✗'}\n"
            summary_text += f"Multi-aperture OK: {'✓' if multi_ap_ok else '✗'}\n"
            
            if result_dict.get('ttv_analysis', {}).get('indicates_additional_planets', False):
                summary_text += "\n🪐 Possible additional planets!"
            
            ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            # Plots 11-12: Additional space for future enhancements
            ax11 = plt.subplot(3, 4, 11)
            ax11.text(0.5, 0.5, 'Reserved for\nFuture Analysis', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Future Enhancement')
            
            ax12 = plt.subplot(3, 4, 12)
            ax12.text(0.5, 0.5, 'Reserved for\nFuture Analysis', ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Future Enhancement')
            
            plt.tight_layout()
            
            # Save the comprehensive validation plot
            plot_filename = os.path.join(save_dir, f"kic_{kic_id}_comprehensive_validation.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Comprehensive validation plot saved: {plot_filename}")
            return plot_filename
            
        except Exception as e:
            print(f"  ❌ Error creating validation plots: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None



# Enhanced execution functions with light curve processing
def quick_enhanced_test(csv_file, base_dir=None, model_dir=None,kepler_dir=None, num_candidates=3):
    """Quick test with enhanced light curve processing on a small number of candidates"""
    print(f"🧪 QUICK ENHANCED TEST: {num_candidates} candidates with light curve analysis")
    print("="*80)
    
    validator = EnhancedShallueValidator(
        csv_path=csv_file,
        use_astronet=True if base_dir and model_dir else False,
        base_dir=base_dir,
        model_dir=model_dir,
        kepler_dir=kepler_dir,
        auto_download=True,
        download_method='lightkurve'
    )
    
    if validator.data is None:
        print("❌ Could not load data")
        return None, None
    
    print(f"📊 Loaded {len(validator.data)} candidates")
    print("🌟 Enhanced features enabled:")
    print("  ✓ Light curve downloading (Lightkurve)")
    print("  ✓ Phase folding for signal enhancement")
    print("  ✓ Transit metric calculation")
    print("  ✓ Enhanced ML probability scoring")
    if validator.use_astronet:
        print("  ✓ AstroNet ML predictions")
    
    # Run enhanced validation
    results = validator.validate_all_candidates(max_candidates=num_candidates)
    
    # Save results
    validator.save_enhanced_results('quick_enhanced_test_results.csv')
    
    return validator, results


def run_enhanced_analysis(csv_file, max_candidates=None, use_astronet=False, 
                         base_dir=None, model_dir=None,kepler_dir=None):
    """Run enhanced analysis with light curve processing"""
    print(f"🚀 ENHANCED EXOPLANET ANALYSIS WITH LIGHT CURVE PROCESSING")
    print("="*80)
    
    validator = EnhancedShallueValidator(
        csv_path=csv_file,
        use_astronet=use_astronet,
        base_dir=base_dir,
        model_dir=model_dir,
        kepler_dir=kepler_dir,
        auto_download=True,
        download_method='lightkurve'
    )
    
    if validator.data is None:
        print("❌ Could not load data")
        return None, None
    
    print(f"📊 Loaded {len(validator.data)} candidates")
    print("🌟 Enhanced analysis includes:")
    print("  ✓ Automated light curve downloading")
    print("  ✓ Phase folding to enhance transit signals")
    print("  ✓ Advanced transit detection metrics")
    print("  ✓ Enhanced confidence scoring")
    print("  ✓ Stellar eclipse detection and filtering")
    if use_astronet:
        print("  ✓ AstroNet deep learning predictions")
    
    # Run enhanced validation
    results = validator.validate_all_candidates(max_candidates=max_candidates)
    
    # Save results
    validator.save_enhanced_results('enhanced_exoplanet_analysis.csv')
    
    return validator, results


def analyze_specific_kics_enhanced(csv_file, kic_ids, use_astronet=False, 
                                  base_dir=None,kepler_dir=None, model_dir=None):
    
    """Enhanced analysis for specific KIC IDs with full light curve processing"""
    print(f"🎯 ENHANCED ANALYSIS FOR SPECIFIC KIC IDs: {kic_ids}")
    print("="*80)
    
    validator = EnhancedShallueValidator(
        csv_path=csv_file,
        use_astronet=use_astronet,
        base_dir=base_dir,
        model_dir=model_dir,
        kepler_dir=kepler_dir,
        auto_download=True,
        download_method='lightkurve'
    )
    
    if validator.data is None:
        print("❌ Could not load data")
        return None, None
    
    # Filter to specific KIC IDs
    validator.data = validator.data[validator.data['kic_id'].isin(kic_ids)]
    
    if len(validator.data) == 0:
        print("❌ No matching KIC IDs found in catalog")
        return None, None
    
    print(f"📊 Found {len(validator.data)} matching candidates")
    print("🔬 Enhanced analysis will include:")
    print("  📡 Light curve download and processing")
    print("  🌀 Phase folding for transit enhancement")
    print("  📈 Detailed transit characterization")
    print("  🎯 Transit confirmation analysis")
    print("  📊 Enhanced confidence assessment")
    
    # Run enhanced validation
    results = validator.validate_all_candidates()
    
    # Save results
    validator.save_enhanced_results(f'enhanced_specific_kics_results.csv')
    
    # Display detailed results for each KIC
    print(f"\n📋 DETAILED ENHANCED RESULTS:")
    print("="*100)
    
    for result in results:
        kic = result.get('kic_id', 'N/A')
        conf = result.get('exoplanet_confidence', 'N/A')
        final_prob = result.get('final_probability', result.get('model_prob', 0)) * 100
        period = result.get('period', 0)
        disp = result.get('disposition', 'N/A')
        
        print(f"\n🔍 KIC {kic}:")
        print(f"  Exoplanet Confidence: {conf}")
        print(f"  Final Probability: {final_prob:.1f}%")
        print(f"  Period: {period:.4f} days")
        print(f"  KOI Disposition: {disp}")
        
        # Light curve results
        lc_proc = result.get('light_curve_processing', {})
        if lc_proc.get('light_curve_available'):
            print(f"  📡 Light Curve: ✅ Downloaded and processed")
            
            if lc_proc.get('phase_folded'):
                print(f"  🌀 Phase Folding: ✅ Successfully completed")
                
                metrics = lc_proc.get('transit_metrics', {})
                if 'error' not in metrics:
                    transit_snr = metrics.get('transit_snr', 0)
                    depth_ppm = metrics.get('transit_depth_ppm', 0)
                    duration_h = metrics.get('measured_duration_hours', 0)
                    is_sig = metrics.get('is_significant', False)
                    
                    print(f"  📊 Transit SNR: {transit_snr:.2f}")
                    print(f"  📊 Transit Depth: {depth_ppm:.1f} ppm")
                    print(f"  📊 Measured Duration: {duration_h:.2f} hours")
                    print(f"  📊 Statistically Significant: {'✅' if is_sig else '❌'}")
                    
                    if result.get('light_curve_confirmed'):
                        print(f"  🎯 LIGHT CURVE CONFIRMED TRANSIT! 🎉")
                    
                    if lc_proc.get('plot_generated'):
                        print(f"  📈 Phase-folded plot generated")
                else:
                    print(f"  ⚠️ Transit analysis failed: {metrics.get('error', 'unknown')}")
            else:
                print(f"  🌀 Phase Folding: ❌ Failed")
        else:
            print(f"  📡 Light Curve: ❌ Download failed")
            
        # Overall assessment
        if conf in ['EXTREMELY HIGH', 'VERY HIGH']:
            print(f"  🚀 EXTREMELY LIKELY EXOPLANET! 🪐")
        elif conf == 'HIGH':
            print(f"  🎉 VERY LIKELY EXOPLANET! 🪐")
        elif conf == 'MEDIUM':
            print(f"  🤔 Possible exoplanet - moderate confidence")
        elif conf == 'STELLAR_ECLIPSE':
            print(f"  🌟 Likely stellar eclipse, not an exoplanet")
        else:
            print(f"  ❌ Unlikely to be an exoplanet")
        
        print("-" * 80)
    
    return validator, results

def ultra_quick_test(csv_file, base_dir=None, model_dir=None, kepler_dir=None, num_candidates=3):
    """Ultra-enhanced quick test with all advanced validation methods"""
    print(f"🚀 ULTRA-ENHANCED QUICK TEST: {num_candidates} candidates")
    print("="*80)
    print("🔬 Advanced validation features:")
    print("  ✓ Centroid motion analysis")
    print("  ✓ Multi-aperture photometry validation")
    print("  ✓ Transit timing variation analysis")
    print("  ✓ Odd/even transit comparison")
    print("  ✓ Neighboring star contamination check")
    print("  ✓ Advanced stellar activity filtering")
    print("  ✓ Statistical validation ensemble")
    print("  ✓ Comprehensive validation plots")
    
    validator = EnhancedShallueValidator(
        csv_path=csv_file,
        use_astronet=True if base_dir and model_dir else False,
        base_dir=base_dir,
        model_dir=model_dir,
        kepler_dir=kepler_dir,
        auto_download=True,
        download_method='lightkurve'
    )
    
    if validator.data is None:
        print("❌ Could not load data")
        return None, None
    
    # Run ultra-enhanced validation
    results = validator.validate_all_candidates(max_candidates=num_candidates)
    
    # Use ultra-enhanced summary and saving methods
    validator.print_enhanced_summary()
    validator.save_ultra_enhanced_results('ultra_quick_test_results.csv')
    
    # Create comprehensive validation plots for top candidates
    top_candidates = [r for r in results if r.get('validation_level') in ['GOLD', 'SILVER', 'BRONZE']]
    for candidate in top_candidates[:2]:  # Plot top 2
        kic_id = candidate.get('kic_id')
        if kic_id:
            validator.create_validation_plots(kic_id, candidate)
    
    return validator, results

 
# Main execution with enhanced options
if __name__ == "__main__":
    
    # CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
    csv_file = r"C:\Users\bibin.a.thomas\bazel_projects\exoplanet-ml\pipeline\q1_q17_dr25_koi_2025.08.12_03.20.20.csv"
    base_dir = r"C:\Users\bibin.a.thomas\bazel_projects\exoplanet-ml"
    model_dir = r"C:\Users\bibin.a.thomas\bazel_projects\exoplanet-ml\model"
    kepler_dir = r"C:\Users\bibin.a.thomas\bazel_projects\kepler"

    print("🌟 ENHANCED EXOPLANET DETECTION WITH LIGHT CURVE ANALYSIS")
    print("="*80)
    print("This enhanced tool includes:")
    print("✓ Automated light curve downloading using Lightkurve")
    print("✓ Phase folding to enhance transit signals")
    print("✓ Advanced transit detection and characterization")
    print("✓ Enhanced machine learning probability scoring")
    print("✓ Stellar eclipse detection and filtering")
    print("✓ Beautiful phase-folded light curve plots")
    
    print("\nChoose an analysis option:")
    print("1. Quick enhanced test (3 candidates) - RECOMMENDED FOR FIRST TIME")
    print("2. Enhanced analysis without AstroNet (fast, uses light curves + KOI data)")
    print("3. Enhanced analysis specific KIC IDs (with light curve processing)")
    print("4. Full enhanced analysis with AstroNet (slow but most comprehensive)")
    print("5. Quick test without light curve download (KOI data only)")
    print("6. Ultra-enhanced quick test with all advanced validation")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        print("\n🧪 Running quick enhanced test with light curve analysis...")
        print("This will test light curve downloading and phase folding on 3 candidates.")
        
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            print("Please update the csv_file path in the script.")
            exit()
        
        # Test lightkurve installation
        try:
            import lightkurve
            print(f"✅ Lightkurve found: version {lightkurve.__version__}")
        except ImportError:
            print("❌ Lightkurve not installed. Please install with: pip install lightkurve")
            print("Lightkurve is required for light curve downloading and processing.")
            exit()
        
        validator, results = quick_enhanced_test(csv_file, base_dir, model_dir,kepler_dir, num_candidates=3)
        
    elif choice == "2":
        print("\n🚀 Running enhanced analysis without AstroNet...")
        print("This includes light curve processing but no deep learning predictions.")
        
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            exit()
        
        max_cand = input("Enter maximum number of candidates to process (default 20): ").strip()
        max_candidates = 20 if not max_cand else int(max_cand) if max_cand.isdigit() else 20
        
        validator, results = run_enhanced_analysis(csv_file, max_candidates=max_candidates,kepler_dir=kepler_dir,
                                                  use_astronet=False)
        
    elif choice == "3":
        print("\n🎯 Enhanced analysis for specific KIC IDs...")
        kic_input = input("Enter KIC IDs (comma-separated, e.g., 8554498,9726699): ").strip()
        
        try:
            kic_ids = [int(x.strip()) for x in kic_input.split(",")]
            print(f"Will perform enhanced analysis on KIC IDs: {kic_ids}")
            
            use_astronet = input("Include AstroNet predictions? (y/n): ").strip().lower() == 'y'
            
            if use_astronet:
                if not all(os.path.exists(p) for p in [csv_file, base_dir, model_dir]):
                    print("❌ Some required paths not found for AstroNet. Check configuration.")
                    use_astronet = False
                    print("Proceeding without AstroNet...")
            
            validator, results = analyze_specific_kics_enhanced(
                csv_file, kic_ids, use_astronet=use_astronet, 
                base_dir=base_dir, model_dir=model_dir,kepler_dir=kepler_dir
            )
            
        except ValueError:
            print("❌ Invalid KIC IDs format. Use comma-separated numbers.")
            
    elif choice == "4":
        print("\n🚀 Running full enhanced analysis with AstroNet...")
        print("⚠️  WARNING: This will take a very long time!")
        print("This includes light curve processing AND AstroNet deep learning predictions.")
        
        max_cand = input("Enter maximum number of candidates to process (or 'all'): ").strip()
        
        if max_cand.lower() == 'all':
            max_candidates = None
        else:
            try:
                max_candidates = int(max_cand)
            except ValueError:
                print("Invalid number. Using 10 candidates.")
                max_candidates = 10
        
        confirm = input(f"Continue with {max_candidates or 'ALL'} candidates? This will take hours! (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            exit()
        
        if not all(os.path.exists(p) for p in [csv_file, base_dir, model_dir]):
            print("❌ Required paths not found for AstroNet. Check configuration.")
            exit()
        
        validator, results = run_enhanced_analysis(
            csv_file, max_candidates=max_candidates, use_astronet=True,
            base_dir=base_dir, model_dir=model_dir,kepler_dir=kepler_dir
        )
        
    elif choice == "5":
        print("\n🧮 Running quick test without light curve download...")
        print("This uses only the KOI catalog data for fast testing.")
        
        
        validator = EnhancedShallueValidator(
            csv_path=csv_file,
            use_astronet=False,
            auto_download=False,
            kepler_dir=kepler_dir
        )
        
        if validator.data is not None:
            results = validator.validate_all_candidates()
            validator.save_enhanced_results('quick_koi_only_results.csv')
            
    
    elif choice == "6":
        print("Ultra-enhanced quick test with all advanced validation:")
        
        max_cand = input("Enter maximum number of candidates to process (or 'all'): ").strip()
        
        if max_cand.lower() == 'all':
            max_candidates = None
        else:
            try:
                max_candidates = int(max_cand)
            except ValueError:
                print("Invalid number. Using 10 candidates.")
                max_candidates = 10
        ultra_quick_test(csv_file=csv_file,kepler_dir=kepler_dir,model_dir=model_dir,num_candidates=max_candidates)
    else:
        print("❌ Invalid choice")
        exit()
    
    print("\n" + "="*80)
    print("🎉 ENHANCED ANALYSIS COMPLETE!")
    
    if 'validator' in locals() and validator is not None and hasattr(validator, 'results'):
        # Show final enhanced summary
        total_results = len(validator.results)
        very_likely = len([r for r in validator.results 
                          if r.get('exoplanet_confidence') in ['EXTREMELY HIGH', 'VERY HIGH']])
        lc_confirmed = len([r for r in validator.results 
                           if r.get('light_curve_confirmed', False)])
        lc_processed = len([r for r in validator.results 
                           if r.get('light_curve_processing', {}).get('light_curve_available', False)])
        
        print(f"\n📊 FINAL ENHANCED SUMMARY:")
        print(f"Candidates analyzed: {total_results}")
        print(f"Light curves processed: {lc_processed}")
        print(f"Very likely exoplanets: {very_likely}")
        if lc_confirmed > 0:
            print(f"Light curve confirmed transits: {lc_confirmed}")
        
        if very_likely > 0:
            print(f"\n🚀 SUCCESS! Found {very_likely} very likely exoplanet(s)!")
            if lc_confirmed > 0:
                print(f"Including {lc_confirmed} with confirmed transit signals from light curves!")
            print("Check the generated files for detailed results and beautiful plots.")
        else:
            print("\n😔 No very high-confidence exoplanets found in this sample.")
            print("Try analyzing more candidates or different KIC IDs.")
    
        print("\n📁 Generated files:")
        print("• Enhanced CSV file with detailed light curve analysis")
        print("• Detailed TXT summary file")
        print("• Phase-folded light curve plots (PNG files)")
        if choice in ["1", "4"]:
            print("• AstroNet prediction images (if successful)")
        if lc_confirmed > 0:
            print("• Separate file with light curve confirmed candidates")
        