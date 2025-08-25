import matplotlib.pyplot as plt
from statistics import harmonic_mean
from astropy.timeseries import LombScargle
import astropy.units as u
import numpy as np
import pandas as pd
from glob import glob
import sys
from tqdm import tqdm

'''
Return weighted sample mean and std
http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance
'''
def weighted_moments(series,weights,unbiased=True,harmonic=False):
    if len(series)==1:
        return series[0], 1.0/np.sqrt(weights[0])
    series=np.array(series)
    weights=np.array(weights)
    weightsum=np.sum(weights)
    weightedmean = np.sum(weights*series)/weightsum
    weightedvariance = np.sum(weights*np.power(series-weightedmean,2))
    if harmonic:
        return weightedmean, harmonic_mean(1.0/weights)
    elif unbiased:
        weightsquaredsum=np.sum(np.power(weights,2))
        return weightedmean, np.sqrt(weightedvariance * weightsum / (weightsum**2 - weightsquaredsum))
    else:
        return weightedmean, np.sqrt(weightedvariance / weightsum)

def weighted_median(series, weights):
    inds = np.argsort(series)
    series = series[inds]
    weights = weights[inds]
    weights /= np.sum(weights)

    counter = 0
    for i, weight in enumerate(weights):
        counter += weight
        if counter >= 0.5:
            break
    return series[i]

PSR_name: str = sys.argv[1]
results_dir: str = f"./results/timing_posteriors_frame_tie/{PSR_name}"
results_files = glob(f"{results_dir}/*results.npy")
n_timing_solutions = len(results_files)
print("Number of timing solutions: " + str(n_timing_solutions))

# Array set to zero but to be replaced with the posteriors
posteriors_arr = np.zeros(n_timing_solutions, dtype=float)

# Try the first file:
results = pd.read_pickle(results_files[0:1])
res_diffs = results.residuals_diff[0]
res_diff_nominal_values = np.asarray([res_diffs.nominal_value for x in res_diffs])
epochs = results.res_epochs[0]

# Lomb-Scargle Periodogram
frequencies, powers = LombScargle(epochs * u.day, res_diff_nominal_values * u.us).autopower()
powers_arr = np.empty((n_timing_solutions, len(powers)), dtype=float)
n_freqs = len(frequencies)

for i, file in tqdm(enumerate(results_files)):

    # Extract results
    results = pd.read_pickle(file)
    res_diffs = results.residuals_diff[0]
    res_diff_nominal_values = np.asarray([res_diffs.nominal_value for x in res_diffs])
    epochs = results.res_epochs[0]
    posteriors_arr[i] = results.posterior[0]

    # Lomb-Scargle Periodogram
    freqs, powers_arr[i, :] = LombScargle(epochs * u.day, res_diff_nominal_values * u.us).autopower()
    if len(freqs) != n_freqs:
        sys.exit("Error in the number of frequencies")

# Normalize the posteriors
posteriors_arr /= np.amax(posteriors_arr)
print("Normalized posteriors: " + str(posteriors_arr))

# Calculate the weighted mean of the periodograms per frequency
weightedmean_arr = np.empty(n_freqs, dtype=float)
std_dev_arr = np.empty(n_freqs, dtype=float)

for i in len(frequencies):
    weightedmean_arr[i], std_dev_arr[i] = weighted_moments(series=powers_arr[:, i], weights=posteriors_arr[i])

# Plot the results
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 14)) #, gridspec_kw = {'wspace':0, 'hspace':0})
axs.plot(frequencies, weightedmean_arr)
axs.axvline(x=1.0/((1 * u.yr).to(u.day)).value, color='C1', linestyle='--', lw=2.0, label="$(\mathrm{1~year})^{-1}$")
axs.axvline(x=1.0/((0.5 * u.yr).to(u.day)).value, color='C2', linestyle='--', lw=2.0, label="$(\mathrm{6~months})^{-1}$")
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel('Frequency [1/day]')
axs.set_ylabel('Log(Power) [$\mathrm{\mu s^2}$/day]')
axs.legend(loc='best')
axs.grid()


plt.tight_layout()
plt.savefig(results_dir + f"/{PSR_name}_periodogram.pdf")
plt.show()


