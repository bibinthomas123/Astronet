# Run this as one cell in Jupyter.
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import medfilt
from scipy.optimize import least_squares

# If you need batman: pip install batman-package
import batman

# -------------------------
# Target & ephemeris
# -------------------------
target_id = "KIC 8013419"   # change to KIC 8260218 or KIC 8013419 as needed
period = 12.73262407       # days
t0 = 176.939054          # BKJD
dur_hr = 2.2083           # hours (from table)
dur_d = dur_hr / 24.0
dur_phase = dur_d / period

# -------------------------
# Download and basic preprocessing
# -------------------------
sr = lk.search_lightcurve(target_id, mission="Kepler", cadence="long")
lc = sr.download_all().stitch().remove_nans()

# prefer pdcsap
if "pdcsap_flux" in lc.columns:
    lc = lc.remove_nans().normalize()
else:
    lc = lc.remove_nans()

# create transit epochs within the lc time span
t_start = lc.time.min().value
t_stop  = lc.time.max().value

# compute integer transit numbers that fall in the time range
n0 = int(np.floor((t_start - t0) / period)) - 1
n1 = int(np.ceil((t_stop - t0) / period)) + 1
epochs = np.arange(n0, n1 + 1)
t_transits = t0 + epochs * period

# -------------------------
# Extract and stack individual transits
# -------------------------
half_window = max(0.08, dur_phase * 3)   # ± this in phase (tweakable)
time_list = []
flux_list = []
n_transits_used = 0
points_per_transit = []

for tmid in t_transits:
    # select a small window around each transit in absolute time
    mask = (lc.time.value >= tmid - half_window * period) & (lc.time.value <= tmid + half_window * period)
    sub = lc[mask]
    if len(sub.time) < 5:
        continue  # skip transit if too few points
    # detrend locally with a low-order polynomial or median, masking the expected transit
    rel_time = sub.time.value - tmid
    # mask central region when fitting baseline
    mask_transit = np.abs(rel_time) < (dur_d * 0.8)  # mask most of the transit
    # polynomial baseline using out-of-transit points
    if np.sum(~mask_transit) >= 5:
        coefs = np.polyfit(rel_time[~mask_transit], sub.flux[~mask_transit], 2)
        baseline = np.polyval(coefs, rel_time)
    else:
        baseline = np.median(sub.flux) * np.ones_like(rel_time)
    norm_flux = sub.flux / baseline
    # shift to relative time in days and store
    time_list.append(rel_time)        # days relative to tmid
    flux_list.append(norm_flux)
    n_transits_used += 1
    points_per_transit.append(len(rel_time))

print(f"Found {n_transits_used} usable transits, median points/transit = {np.median(points_per_transit):.1f}")

if n_transits_used == 0:
    raise RuntimeError("No usable transits found in the LC range. Check ephemeris and lc availability.")

# -------------------------
# Build a stacked time series by concatenating and then re-binning on a fine grid
# -------------------------
# choose a fine grid around transit: e.g., ± 1.5 * duration in days, with fine sampling (~5-10 min)
stack_half_hours = 1.5 * dur_hr  # hours
stack_half_days = stack_half_hours / 24.0
fine_dt = 5.0 / 60.0 / 24.0   # 5 minutes in days

stack_time_grid = np.arange(-stack_half_days, stack_half_days + fine_dt, fine_dt)
stack_flux_accum = np.zeros_like(stack_time_grid)
stack_flux_count = np.zeros_like(stack_time_grid)

# For each transit, rebin its points onto the fine grid via nearest-bin accumulate (simple stack)
for rel_time, rel_flux in zip(time_list, flux_list):
    # for each point, find index in grid
    idx = np.searchsorted(stack_time_grid, rel_time)
    # idx may be at boundaries: adjust
    idx[idx == len(stack_time_grid)] = len(stack_time_grid) - 1
    # add flux to bin (use simple mean later)
    for i_pt, i_bin in enumerate(idx):
        stack_flux_accum[i_bin] += rel_flux[i_pt]
        stack_flux_count[i_bin] += 1

# compute stacked average flux (ignore empty bins)
stack_mean = np.full_like(stack_time_grid, np.nan, dtype=float)
nonzero = stack_flux_count > 0
stack_mean[nonzero] = stack_flux_accum[nonzero] / stack_flux_count[nonzero]

# convert stacked time to phase for plotting consistency
stack_phase = stack_time_grid / period

# -------------------------
# Bin the stacked curve to smooth (phase units)
# -------------------------
# choose bin size to get many bins across the transit width
desired_bins_across_transit = 40
bin_phase = max(0.0003, dur_phase / desired_bins_across_transit)
# build bins
bins = np.arange(stack_phase.min(), stack_phase.max() + bin_phase, bin_phase)
bin_centers = (bins[:-1] + bins[1:])/2.0
bin_mean = np.zeros_like(bin_centers)
bin_n = np.zeros_like(bin_centers)

for i in range(len(bin_centers)):
    mask = (stack_phase >= bins[i]) & (stack_phase < bins[i+1]) & (~np.isnan(stack_mean))
    vals = stack_mean[mask]
    if len(vals) > 0:
        bin_mean[i] = np.nanmean(vals)
        bin_n[i] = len(vals)
    else:
        bin_mean[i] = np.nan

# -------------------------
# Smoothed median curve
# -------------------------
# median filter on the binned mean (skip NaNs)
valid = ~np.isnan(bin_mean)
bin_phase_valid = bin_centers[valid]
bin_mean_valid = bin_mean[valid]
ksize = 9
if ksize % 2 == 0:
    ksize += 1
smooth = medfilt(bin_mean_valid, kernel_size=ksize)

# -------------------------
# Diagnostics: how many stacked points near mid-transit?
# -------------------------
near_mask = np.abs(stack_phase) < (dur_phase*0.15)
print("Total stacked bins within small central window:", np.sum(bin_n[(bin_centers >= -dur_phase*0.15) & (bin_centers <= dur_phase*0.15)]))

# -------------------------
# Transit model via batman (least-squares fit)
# We'll fit for rp (Rp/Rs), a/Rs and inclination; keep period & t0 fixed.
# Use simple starting guesses from KOI: rp ~ sqrt(depth), a/R from duration rough estimate.
# -------------------------
# Estimate starting rp from observed stacked depth
obs_depth = 1.0 - np.nanmin(bin_mean_valid)
rp0 = np.sqrt(max(obs_depth, 1e-6))
# rough a/Rs from duration: dur ~ (P/pi) * (Rs/a) for central transit => a/R ~ P/(pi*dur)
a0 = (period) / (np.pi * dur_d)
inc0 = 89.0  # degrees, assume near edge-on

# set up batman params
def batman_model(phase_array, rp, a, inc_deg, u1=0.3, u2=0.1):
    # phase -> time relative to t0
    t = phase_array * period
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = float(period)
    params.rp = float(rp)
    params.a = float(a)
    params.inc = float(inc_deg)
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    # use supersample to account for long cadence smearing
    m = batman.TransitModel(params, t, supersample_factor=7, exp_time=29.4244/60.0/24.0)  # Kepler LC ~29.4244 min
    return m.light_curve(params)

# residuals function for least squares
def residuals(x, phase_obs, flux_obs):
    rp, a, inc = x
    model = batman_model(phase_obs, rp, a, inc)
    return (model - flux_obs)

# prepare arrays for fitting: use bin_phase_valid & bin_mean_valid
x0 = [rp0, a0, inc0]
# bounds: rp in [1e-4, 0.5], a in [1.1, 1000], inc in [60, 90]
res = least_squares(residuals, x0, args=(bin_phase_valid, bin_mean_valid),
                    bounds=([1e-4, 1.1, 60.0], [0.5, 200.0, 90.0]),
                    xtol=1e-8, ftol=1e-8, max_nfev=2000)

rp_fit, a_fit, inc_fit = res.x
print("Fitted params: rp/Rs = %.4f, a/Rs = %.2f, inc = %.2f deg" % (rp_fit, a_fit, inc_fit))

# compute model on a dense phase grid for plotting
dense_phase = np.linspace(-stack_half_days/period, stack_half_days/period, 2000)
model_dense = batman_model(dense_phase, rp_fit, a_fit, inc_fit)
# -------------------------
# Compute global folded LC
# -------------------------
fold = lc.fold(period=period, epoch_time=t0)
phase_global = fold.phase.value
flux_global = fold.flux.value

# compute batman model on global phase grid
phase_dense_global = np.linspace(-0.5, 0.5, 4000)
model_global = batman_model(phase_dense_global, rp_fit, a_fit, inc_fit)

# -------------------------
# PLOT: Global + Local (stacked)
# -------------------------
# -------------------------
# PLOT 1: Global Folded LC
# -------------------------
plt.figure(figsize=(9,5))
plt.scatter(phase_global, flux_global, s=3, color="black", alpha=0.4, label="Folded LC")
# plt.plot(phase_dense_global, model_global, color="tab:red", lw=2, label="Batman model")
plt.axvline(0, color="red", linestyle="--", lw=1)
plt.xlim(-0.5, 0.5)
plt.ylim(0.980, 1.010)
plt.xlabel("Orbital Phase")
plt.ylabel("Normalized Flux")
plt.title(f"Global Folded Transit of {target_id}")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# PLOT 2: Local Stacked Transit Zoom
# -------------------------
plt.figure(figsize=(9,5))
plt.scatter(stack_phase, stack_mean, s=6, color="lightgray", alpha=0.6, label="Stacked points")
plt.scatter(bin_phase_valid, bin_mean_valid, s=30, color="k", zorder=4, label="Binned stacked")
plt.plot(bin_phase_valid, smooth, color="tab:blue", lw=2.0, label="Smoothed (median)")
# plt.plot(dense_phase, model_dense, color="tab:red", lw=2.0, label="Batman model fit")
plt.axvline(0, color="red", linestyle="--", lw=1, alpha=0.8)
plt.xlim(-stack_half_days/period, stack_half_days/period)
depth_seen = 1.0 - np.nanmin(bin_mean_valid)
plt.ylim(0.97, 1.01)
plt.xlabel("Phase")
plt.ylabel("Normalized Flux")
plt.title(f" {target_id} Transit Zoom (N_transits={n_transits_used})")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
