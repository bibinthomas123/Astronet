import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class TransitValidator:
    def __init__(self, kic_id, period, epoch=None):
        self.kic_id = kic_id
        self.period = period
        self.epoch = epoch
        self.lc = None
        self.folded_lc = None
        
    def download_data(self):
        """Download Kepler light curve data"""
        try:
            search_result = lk.search_lightcurve(f'KIC {self.kic_id}', mission='Kepler')
            self.lc = search_result.download_all().stitch()
            print(f"Downloaded data for KIC {self.kic_id}")
            return True
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def fold_lightcurve(self):
        """Fold light curve on given period"""
        if self.lc is None:
            print("No data available. Run download_data() first.")
            return False
            
        # Remove outliers and flatten
        clean_lc = self.lc.remove_outliers(sigma=3).flatten()
        
        # Fold on period
        if self.epoch:
            self.folded_lc = clean_lc.fold(period=self.period, epoch_time=self.epoch)
        else:
            self.folded_lc = clean_lc.fold(period=self.period)
            
        return True
    
    def analyze_transit_shape(self):
        """Analyze transit shape to distinguish U-shaped vs V-shaped transits"""
        if self.folded_lc is None:
            self.fold_lightcurve()
        
        # Extract transit data
        transit_mask = (np.abs(self.folded_lc.phase.value) < 0.02)
        if np.sum(transit_mask) < 10:
            print("Not enough transit data points")
            return None
            
        # Convert to numpy arrays to avoid TimeDelta issues
        transit_phase = self.folded_lc.phase.value[transit_mask]
        transit_flux = self.folded_lc.flux.value[transit_mask]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(transit_phase) | np.isnan(transit_flux))
        transit_phase = transit_phase[valid_mask]
        transit_flux = transit_flux[valid_mask]
        
        if len(transit_phase) < 10:
            print("Not enough valid transit data points")
            return None
        
        # Sort by phase
        sort_idx = np.argsort(transit_phase)
        transit_phase = transit_phase[sort_idx]
        transit_flux = transit_flux[sort_idx]
        
        # Find transit parameters
        baseline = np.median(transit_flux[(np.abs(transit_phase) > 0.015)])
        min_flux = np.min(transit_flux)
        transit_depth = baseline - min_flux
        min_phase = transit_phase[np.argmin(transit_flux)]
        
        # Calculate transit metrics for shape analysis
        shape_metrics = self.calculate_shape_metrics(transit_phase, transit_flux, baseline)
        
        return {
            'phase': transit_phase,
            'flux': transit_flux,
            'baseline': baseline,
            'min_flux': min_flux,
            'depth': transit_depth,
            'min_phase': min_phase,
            'shape_metrics': shape_metrics
        }
    
    def calculate_shape_metrics(self, phase, flux, baseline):
        """Calculate metrics to distinguish U-shaped from V-shaped transits"""
        # Normalize flux relative to baseline
        norm_flux = (flux - baseline) / (np.min(flux) - baseline)
        
        # 1. Flat bottom ratio - measure how much of transit is at minimum depth
        min_flux = np.min(norm_flux)
        flat_threshold = min_flux + 0.1 * (1 - min_flux)  # 10% above minimum
        flat_bottom_fraction = np.sum(norm_flux <= flat_threshold) / len(norm_flux)
        
        # 2. Ingress/egress symmetry
        min_idx = np.argmin(norm_flux)
        ingress_phase = phase[:min_idx+1]
        egress_phase = phase[min_idx:]
        
        # Calculate slopes during ingress and egress
        if len(ingress_phase) > 3 and len(egress_phase) > 3:
            ingress_slope = np.abs(np.polyfit(ingress_phase, norm_flux[:min_idx+1], 1)[0])
            egress_slope = np.abs(np.polyfit(egress_phase, norm_flux[min_idx:], 1)[0])
            slope_ratio = min(ingress_slope, egress_slope) / max(ingress_slope, egress_slope)
        else:
            slope_ratio = 1.0
        
        # 3. Transit curvature at bottom
        # Look at central 40% of transit
        center_mask = np.abs(phase - phase[min_idx]) < 0.4 * (np.max(np.abs(phase)))
        if np.sum(center_mask) > 5:
            center_flux = norm_flux[center_mask]
            curvature = np.std(center_flux)  # Less curvature = flatter bottom = more U-shaped
        else:
            curvature = 1.0
        
        # 4. Second derivative test (for smoothness)
        if len(phase) > 5:
            second_deriv = np.gradient(np.gradient(norm_flux))
            smoothness = 1.0 / (1.0 + np.std(second_deriv))  # Higher = smoother = more U-shaped
        else:
            smoothness = 0.5
        
        return {
            'flat_bottom_fraction': flat_bottom_fraction,
            'slope_ratio': slope_ratio,
            'curvature': curvature,
            'smoothness': smoothness
        }
    
    def classify_transit_shape(self, shape_metrics):
        """Classify transit as U-shaped or V-shaped based on metrics"""
        if shape_metrics is None:
            return "Unknown"
        
        # Scoring system
        u_score = 0
        
        # Flat bottom (U-shaped transits have flatter bottoms)
        if shape_metrics['flat_bottom_fraction'] > 0.3:
            u_score += 2
        elif shape_metrics['flat_bottom_fraction'] > 0.2:
            u_score += 1
        
        # Symmetry (U-shaped transits are more symmetric)
        if shape_metrics['slope_ratio'] > 0.8:
            u_score += 1
        
        # Low curvature at bottom
        if shape_metrics['curvature'] < 0.1:
            u_score += 2
        elif shape_metrics['curvature'] < 0.2:
            u_score += 1
        
        # Smoothness
        if shape_metrics['smoothness'] > 0.7:
            u_score += 1
        
        # Classification
        if u_score >= 4:
            return "U-shaped (Planet-like)"
        elif u_score >= 2:
            return "Possibly U-shaped"
        else:
            return "V-shaped (Binary-like)"
    
    def plot_transit_shape(self):
        """Plot folded light curve with enhanced transit shape analysis"""
        transit_data = self.analyze_transit_shape()
        
        if transit_data is None:
            print("Could not analyze transit shape")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Full phase plot
        # Convert phase to float values for plotting
        full_phase = self.folded_lc.phase.value
        full_flux = self.folded_lc.flux.value
        valid_full = ~(np.isnan(full_phase) | np.isnan(full_flux))
        
        ax1.scatter(full_phase[valid_full], full_flux[valid_full], s=0.5, alpha=0.6, c='blue')
        ax1.set_title(f'KIC {self.kic_id} - Full Phase')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Normalized Flux')
        ax1.grid(True, alpha=0.3)
        
        # Zoomed transit with shape analysis
        ax2.scatter(transit_data['phase'], transit_data['flux'], s=3, alpha=0.8, c='red', label='Transit data')
        ax2.axhline(transit_data['baseline'], color='green', linestyle='--', alpha=0.7, label='Baseline')
        ax2.axhline(transit_data['min_flux'], color='orange', linestyle='--', alpha=0.7, label='Minimum')
        
        # Add shape classification
        shape_class = self.classify_transit_shape(transit_data['shape_metrics'])
        ax2.set_title(f'Transit Shape: {shape_class}')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Normalized Flux')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Shape metrics visualization
        metrics = transit_data['shape_metrics']
        metric_names = ['Flat Bottom', 'Slope Ratio', 'Low Curvature', 'Smoothness']
        metric_values = [
            metrics['flat_bottom_fraction'],
            metrics['slope_ratio'],
            1 - metrics['curvature'],  # Invert so higher = more U-shaped
            metrics['smoothness']
        ]
        
        colors = ['green' if v > 0.5 else 'red' for v in metric_values]
        bars = ax3.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax3.set_title('U-Shape Indicators')
        ax3.set_ylabel('Score (higher = more U-shaped)')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add threshold line
        ax3.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed assessment
        print(f"Transit depth: {transit_data['depth']:.4f} ({transit_data['depth']*100:.2f}%)")
        print(f"Shape classification: {shape_class}")
        print("Shape metrics:")
        print(f"  - Flat bottom fraction: {metrics['flat_bottom_fraction']:.3f}")
        print(f"  - Ingress/egress symmetry: {metrics['slope_ratio']:.3f}")
        print(f"  - Bottom curvature: {metrics['curvature']:.3f} (lower = flatter)")
        print(f"  - Smoothness: {metrics['smoothness']:.3f}")
        
        if "U-shaped" in shape_class:
            print("✓ Transit shape consistent with planet")
        else:
            print("⚠️  Transit shape may indicate eclipsing binary")
        
        return transit_data
    
    def check_secondary_eclipse(self):
        """Check for secondary eclipses at phase 0.5"""
        if self.folded_lc is None:
            self.fold_lightcurve()
        
        # Convert to numpy arrays
        phase_vals = self.folded_lc.phase.value
        flux_vals = self.folded_lc.flux.value
        
        # Extract data around phase 0.5 (secondary eclipse)
        secondary_mask = (np.abs(phase_vals - 0.5) < 0.02) | (np.abs(phase_vals + 0.5) < 0.02)
        secondary_flux = flux_vals[secondary_mask]
        
        # Extract baseline (out of transit)
        baseline_mask = (np.abs(phase_vals) > 0.1) & (np.abs(phase_vals) < 0.4)
        baseline_flux = flux_vals[baseline_mask]
        
        # Remove NaN values
        secondary_flux = secondary_flux[~np.isnan(secondary_flux)]
        baseline_flux = baseline_flux[~np.isnan(baseline_flux)]
        
        if len(secondary_flux) > 10 and len(baseline_flux) > 50:
            secondary_depth = np.nanmedian(baseline_flux) - np.nanmedian(secondary_flux)
            secondary_std = np.nanstd(secondary_flux) / np.sqrt(len(secondary_flux))
            
            print(f"Secondary eclipse depth: {secondary_depth:.6f} ± {secondary_std:.6f}")
            print(f"Secondary eclipse depth: {secondary_depth*100:.3f}% ± {secondary_std*100:.3f}%")
            
            # Statistical significance
            if secondary_std > 0:
                t_stat = secondary_depth / secondary_std
                print(f"Statistical significance: {t_stat:.1f}σ")
                
                if t_stat > 3:
                    print("⚠️  SIGNIFICANT SECONDARY ECLIPSE DETECTED - Likely eclipsing binary")
                elif t_stat > 2:
                    print("⚠️  Possible secondary eclipse - needs investigation")
                else:
                    print("✓ No significant secondary eclipse detected")
                    
                return secondary_depth, secondary_std, t_stat
        
        print("Insufficient data for secondary eclipse analysis")
        return 0, 0, 0
    
    def check_transit_consistency(self):
        """Check individual transit depths for consistency"""
        if self.lc is None:
            print("No data available.")
            return []
            
        # Find individual transits
        clean_lc = self.lc.remove_outliers(sigma=3).flatten()
        
        # Calculate transit times
        if self.epoch:
            transit_times = [self.epoch + i * self.period for i in range(-10, 11)]
        else:
            # Use middle of dataset as reference
            mid_time = np.median(clean_lc.time.value)
            transit_times = [mid_time + i * self.period for i in range(-10, 11)]
        
        # Extract individual transits
        transit_depths = []
        transit_numbers = []
        
        for i, t_time in enumerate(transit_times):
            mask = (np.abs(clean_lc.time.value - t_time) < 0.3)  # ±0.3 days around transit
            if np.sum(mask) > 20:  # Need enough points
                transit_data = clean_lc[mask]
                
                # Calculate depth
                baseline = np.percentile(transit_data.flux.value, 90)  # Top 10% as baseline
                minimum = np.min(transit_data.flux.value)
                depth = baseline - minimum
                
                if depth > 0:  # Valid transit
                    transit_depths.append(depth)
                    transit_numbers.append(i)
        
        if len(transit_depths) > 3:
            depths = np.array(transit_depths)
            mean_depth = np.mean(depths)
            std_depth = np.std(depths)
            
            print(f"Found {len(transit_depths)} individual transits")
            print(f"Mean transit depth: {mean_depth:.6f} ± {std_depth:.6f}")
            print(f"Relative scatter: {(std_depth/mean_depth)*100:.1f}%")
            
            # Plot individual transit depths
            plt.figure(figsize=(10, 6))
            plt.scatter(transit_numbers, depths, alpha=0.7)
            plt.axhline(mean_depth, color='red', linestyle='--', label=f'Mean: {mean_depth:.6f}')
            plt.axhline(mean_depth + 2*std_depth, color='orange', linestyle=':', alpha=0.7, label='±2σ')
            plt.axhline(mean_depth - 2*std_depth, color='orange', linestyle=':', alpha=0.7)
            plt.xlabel('Transit Number')
            plt.ylabel('Transit Depth')
            plt.title(f'KIC {self.kic_id} - Individual Transit Depths')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Assessment
            if (std_depth/mean_depth) < 0.05:
                print("✓ Transit depths are consistent")
            elif (std_depth/mean_depth) < 0.1:
                print("⚠️  Moderate scatter in transit depths")
            else:
                print("⚠️  Large scatter in transit depths - possible false positive")
                
        return transit_depths
    
    def check_odd_even_variation(self):
        """Check for odd-even transit variations"""
        transit_depths = self.check_transit_consistency()
        
        if len(transit_depths) > 6:
            depths = np.array(transit_depths)
            odd_depths = depths[::2]  # Every other starting from 0
            even_depths = depths[1::2]  # Every other starting from 1
            
            # Make arrays same length
            min_len = min(len(odd_depths), len(even_depths))
            odd_depths = odd_depths[:min_len]
            even_depths = even_depths[:min_len]
            
            if len(odd_depths) > 0 and len(even_depths) > 0:
                odd_mean = np.mean(odd_depths)
                even_mean = np.mean(even_depths)
                odd_std = np.std(odd_depths) / np.sqrt(len(odd_depths))
                even_std = np.std(even_depths) / np.sqrt(len(even_depths))
                
                difference = np.abs(odd_mean - even_mean)
                combined_error = np.sqrt(odd_std**2 + even_std**2)
                
                if combined_error > 0:
                    significance = difference / combined_error
                else:
                    significance = 0
                
                print(f"Odd transits mean depth: {odd_mean:.6f} ± {odd_std:.6f}")
                print(f"Even transits mean depth: {even_mean:.6f} ± {even_std:.6f}")
                print(f"Difference: {difference:.6f} ({significance:.1f}σ)")
                
                if significance > 3:
                    print("⚠️  SIGNIFICANT ODD-EVEN VARIATION - Likely false positive")
                elif significance > 2:
                    print("⚠️  Possible odd-even variation - investigate further")
                else:
                    print("✓ No significant odd-even variation")
                    
                return significance
        
        print("Insufficient transits for odd-even analysis")
        return 0
    
    def full_validation(self):
        """Run complete validation suite with enhanced U-shape detection"""
        print(f"=== VALIDATING KIC {self.kic_id} ===")
        print(f"Period: {self.period:.5f} days")
        print()
        
        # Download data
        if not self.download_data():
            return
        
        # 1. Visual inspection with shape analysis
        print("1. TRANSIT SHAPE ANALYSIS:")
        transit_data = self.plot_transit_shape()
        print()
        
        # 2. Secondary eclipse check
        print("2. SECONDARY ECLIPSE CHECK:")
        sec_depth, sec_std, sec_sig = self.check_secondary_eclipse()
        print()
        
        # 3. Transit consistency
        print("3. TRANSIT DEPTH CONSISTENCY:")
        depths = self.check_transit_consistency()
        print()
        
        # 4. Odd-even check
        print("4. ODD-EVEN VARIATION CHECK:")
        odd_even_sig = self.check_odd_even_variation()
        print()
        
        # Final assessment
        print("=== FINAL ASSESSMENT ===")
        issues = []
        good_signs = []
        
        # Check for issues
        if sec_sig > 3:
            issues.append("Significant secondary eclipse detected")
        if odd_even_sig > 3:
            issues.append("Significant odd-even variation")
        
        # CRITICAL: Check transit shape - this should be a major factor
        shape_classification = ""
        if transit_data:
            shape_classification = self.classify_transit_shape(transit_data['shape_metrics'])
            if "V-shaped" in shape_classification:
                issues.append("V-shaped transit profile (binary-like)")
        
        # Check for good signs
        if transit_data and "U-shaped" in shape_classification:
            good_signs.append("U-shaped transit (planet-like)")
        if sec_sig < 2:
            good_signs.append("No significant secondary eclipse")
        if odd_even_sig < 2:
            good_signs.append("Consistent transit depths")
        
        print("Good indicators:")
        for sign in good_signs:
            print(f"  ✓ {sign}")
        
        if len(issues) == 0:
            print("\n✓ PASSED - Strong planet candidate")
        else:
            print("\nIssues found:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
            
            # Major issues that indicate false positive
            major_issues = ["V-shaped transit profile (binary-like)", 
                          "Significant secondary eclipse detected",
                          "Significant odd-even variation"]
            
            has_major_issue = any(any(major in issue for major in major_issues) for issue in issues)
            
            if has_major_issue:
                print("\n❌ REJECTED - Likely false positive (eclipsing binary)")
            else:
                print("\nRequires further investigation")

# Example usage for your top candidates
def validate_top_candidates():
    """Validate your top planet candidates"""
    candidates = [
        (5131180, 14.8518),  # KIC ID, Period
        (8265218, 12.8299),
        (8869680, 7.03378),
        (2579043, 6.31218)
    ]
    
    for kic_id, period in candidates:
        validator = TransitValidator(kic_id, period)
        validator.full_validation()
        print("\n" + "="*60 + "\n")

# Run validation
if __name__ == "__main__":
    # Validate single candidate
    validator = TransitValidator(11446443, 2.47061337) # Example KIC ID and period
    validator.full_validation()
    # validate_top_candidates()  # Your top candidate
    