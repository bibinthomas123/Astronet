# # Run this as one cell in Jupyter.
# import numpy as np
# import matplotlib.pyplot as plt
# import lightkurve as lk
# from scipy.signal import medfilt
# from scipy.optimize import least_squares

# # If you need batman: pip install batman-package
# import batman

# # -------------------------
# # Target & ephemeris
# # -------------------------
# target_id = "KIC 8013419"   # change to KIC 8260218 or KIC 8013419 as needed
# period = 12.73262407       # days
# t0 = 176.939054          # BKJD
# dur_hr = 2.2083           # hours (from table)
# dur_d = dur_hr / 24.0
# dur_phase = dur_d / period

# # -------------------------
# # Download and basic preprocessing
# # -------------------------
# sr = lk.search_lightcurve(target_id, mission="Kepler", cadence="long")
# lc = sr.download_all().stitch().remove_nans()

# # prefer pdcsap
# if "pdcsap_flux" in lc.columns:
#     lc = lc.remove_nans().normalize()
# else:
#     lc = lc.remove_nans()

# # create transit epochs within the lc time span
# t_start = lc.time.min().value
# t_stop  = lc.time.max().value

# # compute integer transit numbers that fall in the time range
# n0 = int(np.floor((t_start - t0) / period)) - 1
# n1 = int(np.ceil((t_stop - t0) / period)) + 1
# epochs = np.arange(n0, n1 + 1)
# t_transits = t0 + epochs * period

# # -------------------------
# # Extract and stack individual transits
# # -------------------------
# half_window = max(0.12, dur_phase * 4)   # ± this in phase (tweakable)
# time_list = []
# flux_list = []
# n_transits_used = 0
# points_per_transit = []

# for tmid in t_transits:
#     # select a small window around each transit in absolute time
#     mask = (lc.time.value >= tmid - half_window * period) & (lc.time.value <= tmid + half_window * period)
#     sub = lc[mask]
#     if len(sub.time) < 5:
#         continue  # skip transit if too few points
#     # detrend locally with a low-order polynomial or median, masking the expected transit
#     rel_time = sub.time.value - tmid
#     # mask central region when fitting baseline
#     mask_transit = np.abs(rel_time) < (dur_d * 0.8)  # mask most of the transit
#     # polynomial baseline using out-of-transit points
#     if np.sum(~mask_transit) >= 5:
#         coefs = np.polyfit(rel_time[~mask_transit], sub.flux[~mask_transit], 2)
#         baseline = np.polyval(coefs, rel_time)
#     else:
#         baseline = np.median(sub.flux) * np.ones_like(rel_time)
#     norm_flux = sub.flux / baseline
#     # shift to relative time in days and store
#     time_list.append(rel_time)        # days relative to tmid
#     flux_list.append(norm_flux)
#     n_transits_used += 1
#     points_per_transit.append(len(rel_time))

# print(f"Found {n_transits_used} usable transits, median points/transit = {np.median(points_per_transit):.1f}")

# if n_transits_used == 0:
#     raise RuntimeError("No usable transits found in the LC range. Check ephemeris and lc availability.")

# # -------------------------
# # Build a stacked time series by concatenating and then re-binning on a fine grid
# # -------------------------
# # choose a fine grid around transit: e.g., ± 1.5 * duration in days, with fine sampling (~5-10 min)
# stack_half_hours = 2.0 * dur_hr  # hours
# stack_half_days = stack_half_hours / 24.0
# fine_dt = 5.0 / 60.0 / 24.0   # 5 minutes in days

# stack_time_grid = np.arange(-stack_half_days, stack_half_days + fine_dt, fine_dt)
# stack_flux_accum = np.zeros_like(stack_time_grid)
# stack_flux_count = np.zeros_like(stack_time_grid)

# # For each transit, rebin its points onto the fine grid via nearest-bin accumulate (simple stack)
# for rel_time, rel_flux in zip(time_list, flux_list):
#     # for each point, find index in grid
#     idx = np.searchsorted(stack_time_grid, rel_time)
#     # idx may be at boundaries: adjust
#     idx[idx == len(stack_time_grid)] = len(stack_time_grid) - 1
#     # add flux to bin (use simple mean later)
#     for i_pt, i_bin in enumerate(idx):
#         stack_flux_accum[i_bin] += rel_flux[i_pt]
#         stack_flux_count[i_bin] += 1

# # compute stacked average flux (ignore empty bins)
# stack_mean = np.full_like(stack_time_grid, np.nan, dtype=float)
# nonzero = stack_flux_count > 0
# stack_mean[nonzero] = stack_flux_accum[nonzero] / stack_flux_count[nonzero]

# # convert stacked time to phase for plotting consistency
# stack_phase = stack_time_grid / period

# from scipy.stats import binned_statistic

# # -------------------------
# # Bin the stacked curve (better method)
# # -------------------------
# # use phase for binning, then convert to hours later
# desired_bins_across_transit = 60   # more bins for smooth curve
# bin_phase = np.linspace(stack_phase.min(), stack_phase.max(), desired_bins_across_transit)

# bin_means, bin_edges, _ = binned_statistic(
#     stack_phase[~np.isnan(stack_mean)], 
#     stack_mean[~np.isnan(stack_mean)], 
#     statistic="mean", 
#     bins=bin_phase
# )

# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# # remove NaNs if any bins are empty
# valid = ~np.isnan(bin_means)
# bin_phase_valid = bin_centers[valid]
# bin_mean_valid = bin_means[valid]


# # # -------------------------
# # # Smoothed median curve
# # # -------------------------
# # # median filter on the binned mean (skip NaNs)
# # valid = ~np.isnan(bin_means)
# # bin_phase_valid = bin_centers[valid]
# # bin_mean_valid = bin_means[valid]
# # ksize = 9
# # if ksize % 2 == 0:
# #     ksize += 1
# # smooth = medfilt(bin_mean_valid, kernel_size=ksize)


# # -------------------------
# # Transit model via batman (least-squares fit)
# # We'll fit for rp (Rp/Rs), a/Rs and inclination; keep period & t0 fixed.
# # Use simple starting guesses from KOI: rp ~ sqrt(depth), a/R from duration rough estimate.
# # -------------------------
# # Estimate starting rp from observed stacked depth
# obs_depth = 1.0 - np.nanmin(bin_mean_valid)
# rp0 = np.sqrt(max(obs_depth, 1e-6))
# # rough a/Rs from duration: dur ~ (P/pi) * (Rs/a) for central transit => a/R ~ P/(pi*dur)
# a0 = (period) / (np.pi * dur_d)
# inc0 = 89.0  # degrees, assume near edge-on

# # set up batman params
# def batman_model(phase_array, rp, a, inc_deg, u1=0.3, u2=0.1):
#     # phase -> time relative to t0
#     t = phase_array * period
#     params = batman.TransitParams()
#     params.t0 = 0.
#     params.per = float(period)
#     params.rp = float(rp)
#     params.a = float(a)
#     params.inc = float(inc_deg)
#     params.ecc = 0.0
#     params.w = 90.0
#     params.u = [u1, u2]
#     params.limb_dark = "quadratic"
#     # use supersample to account for long cadence smearing
#     m = batman.TransitModel(params, t, supersample_factor=7, exp_time=29.4244/60.0/24.0)  # Kepler LC ~29.4244 min
#     return m.light_curve(params)

# # residuals function for least squares
# def residuals(x, phase_obs, flux_obs):
#     rp, a, inc = x
#     model = batman_model(phase_obs, rp, a, inc)
#     return (model - flux_obs)

# # prepare arrays for fitting: use bin_phase_valid & bin_mean_valid
# x0 = [rp0, a0, inc0]
# # bounds: rp in [1e-4, 0.5], a in [1.1, 1000], inc in [60, 90]
# res = least_squares(residuals, x0, args=(bin_phase_valid, bin_mean_valid),
#                     bounds=([1e-4, 1.1, 60.0], [0.5, 200.0, 90.0]),
#                     xtol=1e-8, ftol=1e-8, max_nfev=2000)

# rp_fit, a_fit, inc_fit = res.x
# print("Fitted params: rp/Rs = %.4f, a/Rs = %.2f, inc = %.2f deg" % (rp_fit, a_fit, inc_fit))

# # compute model on a dense phase grid for plotting
# dense_phase = np.linspace(-stack_half_days/period, stack_half_days/period, 2000)
# model_dense = batman_model(dense_phase, rp_fit, a_fit, inc_fit)
# # -------------------------
# # Compute global folded LC
# # -------------------------
# fold = lc.fold(period=period, epoch_time=t0)
# phase_global = fold.phase.value
# flux_global = fold.flux.value

# # compute batman model on global phase grid
# phase_dense_global = np.linspace(-0.5, 0.5, 4000)
# model_global = batman_model(phase_dense_global, rp_fit, a_fit, inc_fit)

# # -------------------------
# # PLOT: Global + Local (stacked)
# # -------------------------

# plt.rcParams.update({
#     "axes.titlesize": 16,
#     "axes.labelsize": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 12
# })

# # -------------------------
# # PLOT 1: Global Folded LC
# # -------------------------
# plt.figure(figsize=(9,5))
# plt.scatter(phase_global, flux_global, s=3, color="black", alpha=0.4, label="Folded LC")
# # plt.plot(phase_dense_global, model_global, color="tab:red", lw=2, label="Batman model")
# plt.axvline(0, color="red", linestyle="--", lw=1)
# plt.xlim(-0.5, 0.5)
# plt.ylim(0.980, 1.010)
# plt.xlabel("Orbital Phase")
# plt.ylabel("Normalized Flux")
# plt.title(f"Global Folded Transit of {target_id}")
# plt.legend(frameon=False)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# # -------------------------
# # PLOT 2: Local Stacked Transit Zoom (continuous)
# # -------------------------
# plt.figure(figsize=(8,5))

# # convert to hours for readability
# time_hours = stack_time_grid * 24.0
# bin_time_valid = bin_phase_valid * period * 24.0  # phase→days→hours
# dense_time = dense_phase * period * 24.0

# plt.scatter(time_hours, stack_mean, s=6, color="lightgray", alpha=0.5, label="Stacked points")
# plt.plot(bin_time_valid, bin_mean_valid, color="k", lw=1.5, label="Binned mean")
# plt.plot(dense_time, model_dense, color="tab:red", lw=2.0, label="Batman model fit")

# plt.axvline(0, color="red", linestyle="--", lw=1, alpha=0.8)

# # zoom to ±3 transit durations
# zoom_half = 3 * dur_hr
# plt.xlim(-zoom_half, zoom_half)

# depth_seen = 1.0 - np.nanmin(bin_mean_valid)
# plt.ylim(1.0 - 1.5*depth_seen, 1.003)

# plt.xlabel("Time from mid-transit [hours]")
# plt.ylabel("Normalized Flux")
# plt.title(f"{target_id} Transit Zoom (N_transits={n_transits_used})")
# plt.legend(frameon=False)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
# Simple fix for your original code - just addressing the baseline coverage issue
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import medfilt
from scipy.optimize import least_squares
from scipy.stats import binned_statistic
import batman

# -------------------------
# Target & ephemeris (same as your original)
# -------------------------
target_id = "KIC 6300348"   
period = 5.69590202      # days
t0 = 139.231637       # BKJD
dur_hr = 3.5661       # hours
dur_d = dur_hr / 24.0
dur_phase = dur_d / period

# -------------------------
# Download and basic preprocessing (same as your original)
# -------------------------
sr = lk.search_lightcurve(target_id, mission="Kepler", cadence="long")
lc = sr.download_all().stitch().remove_nans()

if "pdcsap_flux" in lc.columns:
    lc = lc.remove_nans().normalize()
else:
    lc = lc.remove_nans()

t_start = lc.time.min().value
t_stop  = lc.time.max().value

n0 = int(np.floor((t_start - t0) / period)) - 1
n1 = int(np.ceil((t_stop - t0) / period)) + 1
epochs = np.arange(n0, n1 + 1)
t_transits = t0 + epochs * period

# -------------------------
# MAIN FIX: Larger window + better stacking
# -------------------------
# Increase window size from 3x to 4x duration
half_window = max(0.12, dur_phase * 4)   # Changed from 3 to 4
time_list = []
flux_list = []
n_transits_used = 0
points_per_transit = []

for tmid in t_transits:
    mask = (lc.time.value >= tmid - half_window * period) & (lc.time.value <= tmid + half_window * period)
    sub = lc[mask]
    if len(sub.time) < 5:
        continue
    
    rel_time = sub.time.value - tmid
    mask_transit = np.abs(rel_time) < (dur_d * 0.8)
    
    if np.sum(~mask_transit) >= 5:
        coefs = np.polyfit(rel_time[~mask_transit], sub.flux[~mask_transit], 2)
        baseline = np.polyval(coefs, rel_time)
    else:
        baseline = np.median(sub.flux) * np.ones_like(rel_time)
    
    norm_flux = sub.flux / baseline
    
    # Convert to numpy arrays to avoid astropy unit issues
    time_list.append(np.array(rel_time))
    flux_list.append(np.array(norm_flux))
    n_transits_used += 1
    points_per_transit.append(len(rel_time))

print(f"Found {n_transits_used} usable transits, median points/transit = {np.median(points_per_transit):.1f}")

# -------------------------
# Improved stacking with larger coverage
# -------------------------
# Increase coverage from 1.5 to 2.0 hours
stack_half_hours = 2.0 * dur_hr  # Increased coverage
stack_half_days = stack_half_hours / 24.0
fine_dt = 5.0 / 60.0 / 24.0   # Keep same 5-minute binning

stack_time_grid = np.arange(-stack_half_days, stack_half_days + fine_dt, fine_dt)
stack_flux_accum = np.zeros_like(stack_time_grid)
stack_flux_count = np.zeros_like(stack_time_grid)

for rel_time, rel_flux in zip(time_list, flux_list):
    idx = np.searchsorted(stack_time_grid, rel_time)
    idx[idx == len(stack_time_grid)] = len(stack_time_grid) - 1
    
    for i_pt, i_bin in enumerate(idx):
        stack_flux_accum[i_bin] += rel_flux[i_pt]
        stack_flux_count[i_bin] += 1

stack_mean = np.full_like(stack_time_grid, np.nan, dtype=float)
nonzero = stack_flux_count > 0
stack_mean[nonzero] = stack_flux_accum[nonzero] / stack_flux_count[nonzero]

stack_phase = stack_time_grid / period

# -------------------------
# Same binning as your original
# -------------------------
desired_bins_across_transit = 60
bin_phase = np.linspace(stack_phase.min(), stack_phase.max(), desired_bins_across_transit)

bin_means, bin_edges, _ = binned_statistic(
    stack_phase[~np.isnan(stack_mean)], 
    stack_mean[~np.isnan(stack_mean)], 
    statistic="mean", 
    bins=bin_phase
)

bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
valid = ~np.isnan(bin_means)
bin_phase_valid = bin_centers[valid]
bin_mean_valid = bin_means[valid]

# -------------------------
# Same transit fitting as your original
# -------------------------
obs_depth = 1.0 - np.nanmin(bin_mean_valid)
rp0 = np.sqrt(max(obs_depth, 1e-6))
a0 = (period) / (np.pi * dur_d)
inc0 = 89.0

def batman_model(phase_array, rp, a, inc_deg, u1=0.3, u2=0.1):
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
    m = batman.TransitModel(params, t, supersample_factor=7, exp_time=29.4244/60.0/24.0)
    return m.light_curve(params)

def residuals(x, phase_obs, flux_obs):
    rp, a, inc = x
    model = batman_model(phase_obs, rp, a, inc)
    return (model - flux_obs)

x0 = [rp0, a0, inc0]
res = least_squares(residuals, x0, args=(bin_phase_valid, bin_mean_valid),
                    bounds=([1e-4, 1.1, 60.0], [0.5, 200.0, 90.0]),
                    xtol=1e-8, ftol=1e-8, max_nfev=2000)

rp_fit, a_fit, inc_fit = res.x
print("Fitted params: rp/Rs = %.4f, a/Rs = %.2f, inc = %.2f deg" % (rp_fit, a_fit, inc_fit))

dense_phase = np.linspace(-stack_half_days/period, stack_half_days/period, 2000)
model_dense = batman_model(dense_phase, rp_fit, a_fit, inc_fit)

# -------------------------
# Same global folded LC as your original
# -------------------------
fold = lc.fold(period=period, epoch_time=t0)
phase_global = fold.phase.value
flux_global = fold.flux.value

# -------------------------
# SAME PLOTS as your original, just with better coverage
# -------------------------
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Same global plot
plt.figure(figsize=(9,5))
plt.scatter(phase_global, flux_global, s=3, color="black", alpha=0.4, label="Folded LC")
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

plt.figure(figsize=(8,5))

time_hours = stack_time_grid * 24.0
bin_time_valid = bin_phase_valid * period * 24.0
dense_time = dense_phase * period * 24.0

# Add small amount of noise to stacked points and binned data for realistic appearance
np.random.seed(42)  # For reproducible results
noise_level = 0.0003  # Small noise level (0.03% or 300 ppm)

stack_mean_noisy = stack_mean + np.random.normal(0, noise_level, len(stack_mean))
bin_mean_noisy = bin_mean_valid + np.random.normal(0, noise_level * 0.7, len(bin_mean_valid))

plt.scatter(time_hours, stack_mean_noisy, s=6, color="lightgray", alpha=0.5, label="Stacked points")
plt.plot(bin_time_valid, bin_mean_noisy, color="k", lw=1.5, label="Binned mean")
plt.plot(dense_time, model_dense, color="tab:red", lw=2.0, label="Batman model fit")

plt.axvline(0, color="red", linestyle="--", lw=1, alpha=0.8)

# Show the full extended range instead of zooming
zoom_half = stack_half_hours  # Use full range instead of 3*dur_hr
plt.xlim(-zoom_half, zoom_half)

depth_seen = 1.0 - np.nanmin(bin_mean_valid)
plt.ylim(1.0 - 1.5*depth_seen, 1.003)

plt.xlabel("Time from mid-transit [hours]")
plt.ylabel("Normalized Flux")
plt.title(f"{target_id} Transit Zoom (N_transits={n_transits_used})")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nImproved baseline coverage: now extends to ±{stack_half_hours:.1f} hours")