#!/usr/bin/env python3
"""
Alternative AO Analysis Pipeline using fitsio library
For when astropy.io.fits fails to read files
"""

import fitsio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
from scipy import ndimage
from scipy.optimize import curve_fit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitsioAOPipeline:
    """AO Analysis using fitsio library as alternative to astropy"""
    
    def __init__(self, target_name="KIC212_KOI064", pixel_scale=0.04):
        self.target_name = target_name
        self.pixel_scale = pixel_scale
        self.data = {}
        self.processed_data = {}
    
    def download_files(self):
        """Download the FITS files"""
        files = {
            'u': "https://exofop.ipac.caltech.edu/tess/kepler_files/ubv/ubv_k064_u.fits",
            'b': "https://exofop.ipac.caltech.edu/tess/kepler_files/ubv/ubv_k064_b.fits", 
            'v': "https://exofop.ipac.caltech.edu/tess/kepler_files/ubv/ubv_k064_v.fits"
        }
        
        for band, url in files.items():
            filename = f"ubv_k064_{band}.fits"
            if not os.path.exists(filename):
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded {filename}")
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
    
    def load_with_fitsio(self):
        """Load FITS files using fitsio library"""
        files = ['ubv_k064_u.fits', 'ubv_k064_b.fits', 'ubv_k064_v.fits']
        bands = ['u', 'b', 'v']
        
        for filename, band in zip(files, bands):
            try:
                # Try fitsio first
                fits_file = fitsio.FITS(filename)
                data = fits_file[0].read()  # Read primary HDU
                header = fits_file[0].read_header() if hasattr(fits_file[0], 'read_header') else {}
                fits_file.close()
                
                # Handle different data shapes
                if data.ndim == 3:
                    data = np.median(data, axis=0)
                
                data = np.nan_to_num(data.astype(np.float64))
                background = np.median(data)
                processed = data - background
                
                self.data[band] = data
                self.processed_data[band] = processed
                
                logger.info(f"Successfully loaded {band}-band with fitsio: {data.shape}")
                
            except Exception as e:
                logger.error(f"fitsio failed for {filename}: {e}")
                continue
        
        return len(self.processed_data) > 0
    
    def simple_source_detection(self, data, threshold_factor=5.0):
        """Simple source detection without photutils"""
        # Calculate background statistics
        background = np.median(data)
        noise = np.std(data)
        threshold = background + threshold_factor * noise
        
        # Find sources above threshold
        sources_mask = data > threshold
        
        # Label connected components
        labeled_sources, num_sources = ndimage.label(sources_mask)
        
        sources = []
        for i in range(1, num_sources + 1):
            source_mask = labeled_sources == i
            source_pixels = np.where(source_mask)
            
            if len(source_pixels[0]) < 5:  # Skip tiny sources
                continue
            
            # Calculate centroid
            y_center = np.mean(source_pixels[0])
            x_center = np.mean(source_pixels[1])
            
            # Calculate flux
            flux = np.sum(data[source_mask])
            
            sources.append({
                'x': x_center,
                'y': y_center,
                'flux': flux
            })
        
        return pd.DataFrame(sources)
    
    def analyze_multiband(self):
        """Perform multi-band analysis"""
        all_sources = {}
        
        for band, data in self.processed_data.items():
            sources = self.simple_source_detection(data)
            if len(sources) > 0:
                all_sources[band] = sources
                logger.info(f"{band.upper()}-band: {len(sources)} sources detected")
        
        return all_sources
    
    def create_plots(self, sources_dict):
        """Create diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot images with sources
        for i, (band, data) in enumerate(self.processed_data.items()):
            ax = axes[0, i]
            vmin, vmax = np.percentile(data, [5, 95])
            ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            
            if band in sources_dict:
                sources = sources_dict[band]
                ax.scatter(sources['x'], sources['y'], c='red', s=50, alpha=0.8)
            
            ax.set_title(f'{band.upper()}-band')
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
        
        # Plot source counts
        ax = axes[1, 0]
        band_names = list(sources_dict.keys())
        counts = [len(sources_dict[band]) for band in band_names]
        ax.bar(band_names, counts, color=['purple', 'blue', 'green'])
        ax.set_title('Sources Detected by Band')
        ax.set_ylabel('Number of Sources')
        
        # Plot flux comparison if multiple bands available
        if len(sources_dict) >= 2:
            bands = list(sources_dict.keys())
            ax = axes[1, 1]
            
            # Simple cross-match by position
            band1, band2 = bands[0], bands[1]
            sources1, sources2 = sources_dict[band1], sources_dict[band2]
            
            matched_flux1, matched_flux2 = [], []
            for _, s1 in sources1.iterrows():
                distances = np.sqrt((sources2['x'] - s1['x'])**2 + (sources2['y'] - s1['y'])**2)
                if len(distances) > 0 and distances.min() < 3.0:
                    closest_idx = distances.idxmin()
                    matched_flux1.append(s1['flux'])
                    matched_flux2.append(sources2.iloc[closest_idx]['flux'])
            
            if matched_flux1:
                ax.scatter(matched_flux1, matched_flux2, alpha=0.7)
                ax.set_xlabel(f'{band1.upper()}-band Flux')
                ax.set_ylabel(f'{band2.upper()}-band Flux')
                ax.set_title('Cross-band Flux Comparison')
                ax.loglog()
        
        plt.tight_layout()
        plt.savefig(f'{self.target_name}_fitsio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Install fitsio if needed
def install_fitsio():
    """Install fitsio package"""
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fitsio"])
        logger.info("fitsio installed successfully")
    except Exception as e:
        logger.error(f"Failed to install fitsio: {e}")

# Main execution using fitsio
if __name__ == "__main__":
    pipeline = FitsioAOPipeline()
    
    try:
        # Try to import fitsio
        import fitsio
    except ImportError:
        logger.warning("fitsio not found, attempting to install...")
        install_fitsio()
        import fitsio
    
    pipeline.download_files()
    
    if pipeline.load_with_fitsio():
        sources = pipeline.analyze_multiband()
        pipeline.create_plots(sources)
        print("Analysis completed using fitsio!")
    else:
        print("Failed to load files with fitsio")


# Alternative using PyFITS (older but sometimes more forgiving)
try:
    import pyfits as fits  # Older PyFITS
except ImportError:
    from astropy.io import fits

def load_with_pyfits(filename):
    try:
        hdul = fits.open(filename, ignore_missing_end=True, checksum=False)
        data = hdul[0].data
        return data
    except:
        return None


def read_fits_raw(filename):
    """Read FITS file as raw binary when all else fails"""
    try:
        with open(filename, 'rb') as f:
            # Skip FITS header (simplified)
            header_size = 2880  # Standard FITS header block
            while True:
                block = f.read(header_size)
                if b'END' in block:
                    # Find actual end of header
                    end_pos = block.find(b'END')
                    remaining = header_size - end_pos - 80
                    if remaining > 0:
                        f.seek(-remaining, 1)
                    break
            
            # Read data (you'll need to know the dimensions)
            # This is very basic - real implementation needs header parsing
            data_bytes = f.read()
            
            # Convert to numpy array (assuming float32, adjust as needed)
            data = np.frombuffer(data_bytes, dtype='>f4')  # Big-endian float32
            
            # Reshape based on expected dimensions (you'd need to parse header for this)
            # This is just an example - adjust dimensions as needed
            expected_size = int(np.sqrt(len(data)))
            if expected_size * expected_size == len(data):
                data = data.reshape(expected_size, expected_size)
            
            return data
            
    except Exception as e:
        print(f"Raw reading failed: {e}")
        return None

import subprocess
import tempfile

def convert_fits_with_ds9(fits_file, output_format='numpy'):
    """Use ds9 to convert problematic FITS files"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            # Use ds9 to export as text array
            cmd = f"ds9 {fits_file} -saveimage {tmp.name} -exit"
            subprocess.run(cmd, shell=True, check=True)
            
            # Read the exported data
            data = np.loadtxt(tmp.name)
            os.unlink(tmp.name)
            return data
    except:
        return None


def analyze_without_fits():
    """Alternative approach using web scraping or API calls"""
    # Check if ExoFOP has API access
    # Or try different file formats if available
    
    # Example: Check for other formats
    alternative_urls = [
        "https://exofop.ipac.caltech.edu/tess/kepler_files/png/k064_u.png",  # Images
        "https://exofop.ipac.caltech.edu/tess/kepler_files/csv/k064_data.csv",  # CSV data
    ]
    
    for url in alternative_urls:
        try:
            response = requests.head(url)
            if response.status_code == 200:
                print(f"Alternative format available: {url}")
        except:
            continue


def repair_fits_file(filename):
    """Attempt to repair corrupted FITS file"""
    try:
        # Try with different astropy options
        from astropy.io import fits
        
        # Method 1: Ignore verification
        hdul = fits.open(filename, ignore_missing_simple=True, 
                        ignore_missing_end=True, checksum=False)
        
        # Method 2: Try to fix and resave
        hdul.verify('fix')
        repaired_name = filename.replace('.fits', '_repaired.fits')
        hdul.writeto(repaired_name, overwrite=True)
        
        return repaired_name
        
    except Exception as e:
        print(f"Repair failed: {e}")
        return None
