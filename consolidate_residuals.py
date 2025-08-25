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
posteriors_dir: str = f"./results/timing_posteriors_frame_tie/{PSR_name}"
posterior_files = glob(f"{posteriors_dir}/*posterior.npy")
residual_files = glob(f"{posteriors_dir}/*residuals_diff.npy")

# Array set to zero but to be replaced with the posteriors
nbins: int = np.shape(np.load(residual_files[0:1]))[0]
res_np = np.zeros((len(residual_files), nbins), dtype=object)
posteriors_np = np.zeros(len(posterior_files), dtype=float)

for i, (res_file, post_file) in enumerate(zip(residual_files,posterior_files)):


    # Lomb-Scargle Periodogram
    avg_residuals = np.load(res_file)
    frequency, power = LombScargle(epochs * u.day, avg_residuals * u.us).autopower()



    posteriors_np[i] = np.load(post_file)[-1]

# Normalize the posteriors
posteriors_np /= np.amax(posteriors_np)



# Take the weighted sum of the individual periodograms

# Save the results to a DataFrame
result_df = pd.DataFrame(data=res_np, columns=["POSEPOCH", "RAJ", "DECJ", "PX", "PMRA", "PMDEC", "posterior"]).sort_index()
result_df.to_pickle(f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_residuals.pkl")
