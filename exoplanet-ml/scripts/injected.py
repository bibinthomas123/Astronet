import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

class ModelInjectionTester:
    """
    Test your trained model using planet injection methodology
    """
    
    def __init__(self, model_path, data_root_path):
        """
        Initialize with your trained model and data paths
        
        Parameters:
        model_path: Path to your saved trained model (.h5 or .keras)
        data_root_path: Root path where your Kepler light curves are stored
        """
        self.model_path = model_path
        self.data_root_path = Path(data_root_path)
        self.model = None
        self.injection_results = []
        
        print(f"🤖 Loading trained model from: {model_path}")
        self.load_model()
        
    def load_model(self):
        """Load your trained CNN-BiLSTM-Attention model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"📋 Model input shape: {self.model.input_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def load_kepler_lightcurve(self, kic_id):
        """
        Load Kepler light curve data - adapt this to your data format
        """
        # Adapt this path structure to match your data organization
        kic_folder = self.data_root_path / f"{str(kic_id)[:4]}" / f"{kic_id:09d}"
        
        # Look for light curve files (adapt extensions as needed)
        possible_files = [
            kic_folder / f"kplr{kic_id:09d}_lc.fits",
            kic_folder / f"{kic_id}_lightcurve.csv",
            kic_folder / f"lightcurve_{kic_id}.txt"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return self._load_file(file_path)
        
        print(f"⚠️ No light curve found for KIC {kic_id}")
        return None
    
    def _load_file(self, file_path):
        """Load light curve from file - adapt to your format"""
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.fits':
                # FITS file format
                with fits.open(file_path) as hdul:
                    data = hdul[1].data
                    time = data['TIME']
                    flux = data['PDCSAP_FLUX']
                    flux_err = data['PDCSAP_FLUX_ERR']
                    
            elif file_ext == '.csv':
                # CSV format
                df = pd.read_csv(file_path)
                time = df['time'].values
                flux = df['flux'].values
                flux_err = df['flux_err'].values
                
            elif file_ext == '.txt':
                # Text format
                data = np.loadtxt(file_path)
                time = data[:, 0]
                flux = data[:, 1]
                flux_err = data[:, 2] if data.shape[1] > 2 else np.ones_like(flux) * 0.001
            
            # Clean data
            mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
            return {
                'time': time[mask],
                'flux': flux[mask],
                'flux_err': flux_err[mask]
            }
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def preprocess_for_model(self, time, flux, flux_err):
        """
        Apply the SAME preprocessing as your training pipeline
        CRITICAL: This must match your training preprocessing exactly
        """
        # Remove outliers (adapt sigma clipping threshold as needed)
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        outlier_mask = np.abs(flux - median_flux) < 5 * mad
        
        time = time[outlier_mask]
        flux = flux[outlier_mask]
        flux_err = flux_err[outlier_mask]
        
        # Normalize flux
        flux = flux / np.median(flux) - 1.0
        
        # Detrending (adapt to your method)
        # Option 1: Simple polynomial detrend
        coeffs = np.polyfit(time, flux, deg=3)
        trend = np.polyval(coeffs, time)
        flux = flux - trend
        
        # Option 2: If you use scipy.signal.savgol_filter
        # from scipy.signal import savgol_filter
        # window_length = min(101, len(flux)//10 * 2 + 1)  # Ensure odd
        # flux = flux - savgol_filter(flux, window_length, 3)
        
        # Binning/resampling to fixed length (adapt to your model's expected input)
        target_length = 2001  # Change this to match your model's input size
        
        if len(flux) > target_length:
            # Downsample
            indices = np.linspace(0, len(flux)-1, target_length, dtype=int)
            time = time[indices]
            flux = flux[indices]
            flux_err = flux_err[indices]
        elif len(flux) < target_length:
            # Upsample using interpolation
            f_interp = interpolate.interp1d(time, flux, kind='linear', fill_value='extrapolate')
            fe_interp = interpolate.interp1d(time, flux_err, kind='linear', fill_value='extrapolate')
            
            time_new = np.linspace(time.min(), time.max(), target_length)
            flux = f_interp(time_new)
            flux_err = fe_interp(time_new)
            time = time_new
        
        # Final normalization (adapt to your method)
        flux_std = np.std(flux)
        if flux_std > 0:
            flux = flux / flux_std
        
        return time, flux, flux_err
    
    def inject_synthetic_planet(self, time, flux, period, t0, depth, duration):
        """
        Inject a synthetic planet transit into the light curve
        """
        # Phase fold the time array
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0  # Center around 0
        
        # Convert phase to hours
        phase_hours = phase * period * 24
        
        # Create transit model (simple trapezoidal)
        transit_model = np.ones_like(time)
        ingress_duration = duration * 0.15  # 15% for ingress/egress
        
        # Core transit
        in_core = np.abs(phase_hours) <= (duration/2 - ingress_duration)
        transit_model[in_core] = 1.0 - depth
        
        # Ingress/egress
        in_ingress = (np.abs(phase_hours) > (duration/2 - ingress_duration)) & \
                     (np.abs(phase_hours) <= duration/2)
        
        for i, ph in enumerate(phase_hours):
            if in_ingress[i]:
                fade_factor = (duration/2 - np.abs(ph)) / ingress_duration
                transit_model[i] = 1.0 - depth * fade_factor
        
        # Apply transit (multiplicative)
        injected_flux = flux * transit_model
        
        return injected_flux, transit_model
    
    def generate_test_parameters(self, n_planets=500):
        """
        Generate parameters for injection testing
        Stratified across period ranges to test your model's strengths
        """
        np.random.seed(42)  # Reproducible results
        
        params = []
        
        # Short periods (0.5 - 10 days) - 40% of sample
        n_short = int(0.4 * n_planets)
        for i in range(n_short):
            period = np.random.uniform(0.5, 10)
            radius_earth = np.random.lognormal(0, 0.5) + 0.8  # 0.8-4 Earth radii
            radius_earth = np.clip(radius_earth, 0.8, 4.0)
            
            params.append({
                'injection_id': f'SHORT_{i:04d}',
                'period': period,
                'radius_earth': radius_earth,
                'category': 'short'
            })
        
        # Medium periods (10 - 100 days) - 40% of sample  
        n_medium = int(0.4 * n_planets)
        for i in range(n_medium):
            period = np.random.uniform(10, 100)
            radius_earth = np.random.lognormal(0, 0.6) + 1.0
            radius_earth = np.clip(radius_earth, 1.0, 6.0)
            
            params.append({
                'injection_id': f'MEDIUM_{i:04d}',
                'period': period,
                'radius_earth': radius_earth,
                'category': 'medium'
            })
        
        # Long periods (100 - 400 days) - 20% of sample (your model's strength!)
        n_long = n_planets - n_short - n_medium
        for i in range(n_long):
            period = np.random.uniform(100, 400)
            radius_earth = np.random.lognormal(0.2, 0.4) + 1.5
            radius_earth = np.clip(radius_earth, 1.5, 8.0)
            
            params.append({
                'injection_id': f'LONG_{i:04d}',
                'period': period,
                'radius_earth': radius_earth,
                'category': 'long'
            })
        
        # Calculate physical parameters for each
        for p in params:
            # Assume solar-type star (R* = 1 R_sun)
            stellar_radius = 1.0  # Solar radii
            
            # Transit depth
            depth = (p['radius_earth'] * const.R_earth / (stellar_radius * const.R_sun))**2
            p['transit_depth'] = float(depth.decompose())
            
            # Transit duration (simplified)
            # Duration ≈ (P/π) * sqrt(1-b²) * (R*/a)
            # For circular orbits: a ≈ ((P²*G*M*)/(4π²))^(1/3)
            semi_major = ((p['period'] * u.day)**2 * const.G * const.M_sun / (4 * np.pi**2))**(1/3)
            duration_fraction = (stellar_radius * const.R_sun / semi_major).decompose()
            p['duration'] = float(p['period'] * 24 * duration_fraction / np.pi)  # Hours
            
            # Random phase
            p['t0'] = np.random.uniform(0, p['period'])
        
        return pd.DataFrame(params)
    
    def run_injection_testing(self, clean_kic_list, n_test_planets=500):
        """
        Main function to run injection testing on your trained model
        """
        print(f"🌍 Starting injection testing with {n_test_planets} synthetic planets")
        
        # Generate injection parameters
        injection_params = self.generate_test_parameters(n_test_planets)
        print(f"📊 Generated parameters for {len(injection_params)} planets:")
        print(f"   - Short period (<10d): {len(injection_params[injection_params['category']=='short'])}")
        print(f"   - Medium period (10-100d): {len(injection_params[injection_params['category']=='medium'])}")  
        print(f"   - Long period (>100d): {len(injection_params[injection_params['category']=='long'])}")
        
        results = []
        
        for idx, params in injection_params.iterrows():
            try:
                # Randomly select a clean light curve
                kic_id = np.random.choice(clean_kic_list)
                
                # Load light curve
                lc_data = self.load_kepler_lightcurve(kic_id)
                if lc_data is None:
                    continue
                
                # Inject planet
                injected_flux, transit_model = self.inject_synthetic_planet(
                    lc_data['time'], lc_data['flux'], params
                )
                
                # Preprocess for model (SAME as training)
                processed_time, processed_flux, processed_err = self.preprocess_for_model(
                    lc_data['time'], injected_flux, lc_data['flux_err']
                )
                
                # Reshape for model input (adapt to your model's expected shape)
                model_input = processed_flux.reshape(1, -1, 1)  # (batch, timesteps, features)
                
                # Run prediction
                prediction = self.model.predict(model_input, verbose=0)[0][0]
                
                # Store results
                result = {
                    'injection_id': params['injection_id'],
                    'host_kic': kic_id,
                    'true_period': params['period'],
                    'true_depth': params['transit_depth'],
                    'true_duration': params['duration'],
                    'true_radius': params['radius_earth'],
                    'category': params['category'],
                    'prediction_score': float(prediction),
                    'detected': prediction > 0.5,
                    'true_positive': prediction > 0.5,  # All injections are true planets
                    'false_negative': prediction <= 0.5
                }
                
                results.append(result)
                
                if len(results) % 50 == 0:
                    print(f"📈 Processed {len(results)}/{len(injection_params)} injections")
                    
            except Exception as e:
                print(f"⚠️ Error processing {params['injection_id']}: {e}")
                continue
        
        self.injection_results = pd.DataFrame(results)
        print(f"✅ Injection testing complete! Processed {len(self.injection_results)} planets")
        
        return self.injection_results
    
    def inject_synthetic_planet(self, time, flux, params):
        """Inject planet using your parameters"""
        return self.inject_synthetic_planet_simple(
            time, flux, 
            params['period'], 
            params['t0'],
            params['transit_depth'], 
            params['duration']
        )
    
    def inject_synthetic_planet_simple(self, time, flux, period, t0, depth, duration):
        """Simple planet injection"""
        # Phase calculations
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        phase_hours = phase * period * 24
        
        # Transit model
        transit_model = np.ones_like(time)
        ingress_duration = duration * 0.15
        
        in_core = np.abs(phase_hours) <= (duration/2 - ingress_duration)
        transit_model[in_core] = 1.0 - depth
        
        in_ingress = (np.abs(phase_hours) > (duration/2 - ingress_duration)) & \
                     (np.abs(phase_hours) <= duration/2)
        
        for i, ph in enumerate(phase_hours):
            if in_ingress[i]:
                fade_factor = (duration/2 - np.abs(ph)) / ingress_duration
                transit_model[i] = 1.0 - depth * fade_factor
        
        injected_flux = flux * transit_model
        return injected_flux, transit_model
    
    def calculate_recovery_statistics(self):
        """Calculate detailed recovery statistics"""
        if len(self.injection_results) == 0:
            print("❌ No injection results to analyze!")
            return None
        
        df = self.injection_results
        
        stats = {
            'total_injected': len(df),
            'total_recovered': df['detected'].sum(),
            'overall_recovery_rate': df['detected'].mean(),
            
            # By period category
            'short_period_recovery': df[df['category']=='short']['detected'].mean(),
            'medium_period_recovery': df[df['category']=='medium']['detected'].mean(), 
            'long_period_recovery': df[df['category']=='long']['detected'].mean(),
            
            # By planet size
            'small_planet_recovery': df[df['true_radius'] <= 2]['detected'].mean(),
            'large_planet_recovery': df[df['true_radius'] > 2]['detected'].mean(),
            
            # Confidence statistics
            'mean_confidence': df['prediction_score'].mean(),
            'median_confidence': df['prediction_score'].median(),
            'recovered_mean_confidence': df[df['detected']]['prediction_score'].mean(),
            'missed_mean_confidence': df[~df['detected']]['prediction_score'].mean()
        }
        
        return stats
    
    def plot_recovery_analysis(self):
        """Create recovery analysis plots"""
        if len(self.injection_results) == 0:
            print("❌ No results to plot!")
            return None
            
        df = self.injection_results
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Recovery rate by period
        period_bins = np.logspace(np.log10(0.5), np.log10(400), 15)
        recovery_rates = []
        bin_centers = []
        
        for i in range(len(period_bins)-1):
            mask = (df['true_period'] >= period_bins[i]) & (df['true_period'] < period_bins[i+1])
            if mask.sum() > 0:
                recovery_rates.append(df[mask]['detected'].mean())
                bin_centers.append(np.sqrt(period_bins[i] * period_bins[i+1]))
        
        ax1.semilogx(bin_centers, recovery_rates, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
        ax1.set_xlabel('Orbital Period (days)')
        ax1.set_ylabel('Recovery Rate')
        ax1.set_title('Planet Recovery Rate vs Orbital Period')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        
        # 2. Recovery by category
        categories = ['short', 'medium', 'long']
        recovery_by_cat = [df[df['category']==cat]['detected'].mean() for cat in categories]
        colors = ['blue', 'orange', 'red']
        
        bars = ax2.bar(categories, recovery_by_cat, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Recovery Rate')
        ax2.set_title('Recovery Rate by Period Category')
        ax2.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar, rate in zip(bars, recovery_by_cat):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence score distribution
        recovered = df[df['detected']]
        missed = df[~df['detected']]
        
        ax3.hist(recovered['prediction_score'], bins=20, alpha=0.7, 
                label=f'Recovered ({len(recovered)})', color='green', density=True)
        ax3.hist(missed['prediction_score'], bins=20, alpha=0.7,
                label=f'Missed ({len(missed)})', color='red', density=True)
        ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Detection Threshold')
        ax3.set_xlabel('Model Prediction Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Period vs Planet Size recovery map
        period_grid = np.logspace(np.log10(0.5), np.log10(400), 8)
        radius_grid = np.linspace(0.8, 8, 8)
        
        recovery_grid = np.zeros((len(radius_grid)-1, len(period_grid)-1))
        
        for i in range(len(period_grid)-1):
            for j in range(len(radius_grid)-1):
                mask = (df['true_period'] >= period_grid[i]) & \
                       (df['true_period'] < period_grid[i+1]) & \
                       (df['true_radius'] >= radius_grid[j]) & \
                       (df['true_radius'] < radius_grid[j+1])
                
                if mask.sum() >= 3:  # Need minimum samples
                    recovery_grid[j, i] = df[mask]['detected'].mean()
                else:
                    recovery_grid[j, i] = np.nan
        
        im = ax4.imshow(recovery_grid, aspect='auto', origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1)
        ax4.set_xticks(range(len(period_grid)-1))
        ax4.set_xticklabels([f'{p:.1f}' for p in period_grid[:-1]], rotation=45)
        ax4.set_yticks(range(len(radius_grid)-1))
        ax4.set_yticklabels([f'{r:.1f}' for r in radius_grid[:-1]])
        ax4.set_xlabel('Period (days)')
        ax4.set_ylabel('Radius (Earth Radii)')
        ax4.set_title('Recovery Rate Heatmap')
        plt.colorbar(im, ax=ax4, label='Recovery Rate')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_dir="injection_validation"):
        """Save all results and plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        self.injection_results.to_csv(f"{output_dir}/injection_test_results.csv", index=False)
        
        # Calculate and save statistics
        stats = self.calculate_recovery_statistics()
        
        # Create summary report
        with open(f"{output_dir}/injection_summary.txt", 'w') as f:
            f.write("PLANET INJECTION VALIDATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total planets injected: {stats['total_injected']}\n")
            f.write(f"Total planets recovered: {stats['total_recovered']}\n")
            f.write(f"Overall recovery rate: {stats['overall_recovery_rate']:.1%}\n\n")
            f.write("RECOVERY BY PERIOD CATEGORY:\n")
            f.write(f"Short period (<10d): {stats['short_period_recovery']:.1%}\n")
            f.write(f"Medium period (10-100d): {stats['medium_period_recovery']:.1%}\n") 
            f.write(f"Long period (>100d): {stats['long_period_recovery']:.1%}\n\n")
            f.write("RECOVERY BY PLANET SIZE:\n")
            f.write(f"Small planets (≤2 Re): {stats['small_planet_recovery']:.1%}\n")
            f.write(f"Large planets (>2 Re): {stats['large_planet_recovery']:.1%}\n")
        
        # Create and save plots
        fig = self.plot_recovery_analysis()
        if fig:
            fig.savefig(f"{output_dir}/recovery_analysis.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{output_dir}/recovery_analysis.pdf", bbox_inches='tight')
        
        print(f"💾 Results saved to {output_dir}/")
        return stats

# Main execution function
def main():
    """
    Main function to run injection testing
    ADAPT THESE PATHS TO YOUR SETUP
    """
    
    # CONFIGURE THESE PATHS FOR YOUR SETUP
    MODEL_PATH = "path/to/your/trained_model.h5"  # Your trained model
    DATA_ROOT = "C:/Users/bibin.a.thomas/bazel_projects/kepler"  # Your data root
    
    # List of clean KIC IDs (stars with no known planets)
    # Get these from your training set's negative examples
    CLEAN_KICS = [
        # Add KIC IDs here - stars you used as negative examples in training
        # These should be confirmed non-planet hosts
        # Example format:
        # 1234567, 2345678, 3456789, 4567890, 5678901
    ]
    
    if not CLEAN_KICS:
        print("⚠️ Please add clean KIC IDs to CLEAN_KICS list!")
        print("Use stars from your training set that have no known planets")
        return
    
    # Initialize tester
    tester = ModelInjectionTester(MODEL_PATH, DATA_ROOT)
    
    # Run injection testing
    results = tester.run_injection_testing(CLEAN_KICS, n_test_planets=300)
    
    # Analyze and save results
    stats = tester.save_results("injection_validation_results")
    
    # Print summary
    print("\n" + "="*60)
    print("INJECTION TESTING COMPLETE!")
    print("="*60)
    print(f"Overall Recovery Rate: {stats['overall_recovery_rate']:.1%}")
    print(f"Long Period Recovery: {stats['long_period_recovery']:.1%}")
    print("Check 'injection_validation_results/' folder for detailed results")
    
    return tester, results, stats

if __name__ == "__main__":
    # Run the injection testing
    tester, results, statistics = main()