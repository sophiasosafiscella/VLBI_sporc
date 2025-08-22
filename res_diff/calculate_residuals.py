# Calculate residuals after replacing the astrometric parameters in the timing model with the maximum posterior
# astrometric parameters derived from using VLBI priors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainties import unumpy, ufloat

import pint.fitter
from pint.residuals import Residuals
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from VLBI_utils import epoch_scrunch, replace_params

import os
import sys
import glob

from pint.toa import get_TOAs

if __name__ == "__main__":

    PSR_name: str = "J0030+0451"
#    PSR_name: str = "J2145-0750"

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"../data/NG_15yr_dataset/tim/{PSR_name}_PINT*tim")[0]
    parfile: str = glob.glob(f"../data/NG_15yr_dataset/par/{PSR_name}_PINT*par")[0]

    # Get the nominal timing values
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # Get the TOAs
    toas = get_TOAs(timfile, planets=True, ephem=ec_timing_model.EPHEM.value)

    # ECORR average
    fitter_object = pint.fitter.DownhillGLSFitter(toas, ec_timing_model)
    avg_dict = fitter_object.resids.ecorr_average(use_noise_model=True)
    res_avg = avg_dict['time_resids'].to(u.us).value
    res_avg_errs = avg_dict['errors'].to(u.us).value
    avg_mjds = avg_dict['mjds'].value
    avg_freqs = avg_dict['freqs'].value

    # Average the observations at different frequencies within each time window
    ng15_epochs, ng15_avg_residuals, ng_15_avg_errors = epoch_scrunch(avg_mjds, data=res_avg, errors=res_avg_errs, weighted=True)

    ng15_res = unumpy.uarray(ng15_avg_residuals, ng_15_avg_errors)

    # Load the posteriors
    posteriors_file: str = f"../results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"
    result_df = pd.read_pickle(posteriors_file)

    # Convert PX, PMRA, PMDEC to float
    result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # Find the solution with the highest posterior
    best_sol_idx = result_df['posterior'].idxmax()
    best_sol = result_df.loc[best_sol_idx]

    # Replace the old astrometric values with the maximum posterior ones
    replace_params(eq_timing_model, best_sol)

    # Find the residuals with the new timing solution
    maxpost_fitter_object = pint.fitter.DownhillGLSFitter(toas, eq_timing_model)
    maxpost_avg_dict = maxpost_fitter_object.resids.ecorr_average(use_noise_model=True)
    maxpost_res_avg = maxpost_avg_dict['time_resids'].to(u.us).value
    maxpost_res_avg_errs = maxpost_avg_dict['errors'].to(u.us).value
    maxpost_avg_mjds = maxpost_avg_dict['mjds'].value
    maxpost_avg_freqs = maxpost_avg_dict['freqs'].value

    # Average the observations at different frequencies within each time window
    maxpost_epochs, maxpost_avg_residuals, maxpost_avg_errors = epoch_scrunch(maxpost_avg_mjds, data=maxpost_res_avg, errors=maxpost_res_avg_errs, weighted=True)

    maxpost_res = unumpy.uarray(maxpost_avg_residuals, maxpost_avg_errors)
#    maxpost_res = unumpy.uarray(maxpost_res_avg, maxpost_res_avg_errs)

    # Take the difference
    res_diff = ng15_res - maxpost_res
    res_diff_nominal_values = np.asarray([x.nominal_value for x in res_diff])
    res_diff_std_dev = np.asarray([x.std_dev for x in res_diff])

    # Save the results
    data = pd.DataFrame({'MJD': ng15_epochs, 'Residuals Difference': res_diff_nominal_values, 'Error': res_diff_std_dev})
    data.to_csv(f"./{PSR_name}_res_diff.csv", index=False)

    # Now create the figure
    sns.set(context="paper", style="ticks", font_scale=3.0)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(ng15_epochs, res_diff_nominal_values, yerr=res_diff_std_dev / 10.0, fmt='.', capsize=5, capthick=0.5)

    ax.set_title(PSR_name)#Difference in Pre-Fit Timing Residuals)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residuals difference [$\mathrm{\mu s}$] \n (Re-scaled uncertainties)")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(
#        0.98, 0.95,  # X and Y position in *axes coordinates* (0 to 1)
#        f"Error Median = {median_error:.2f}",  # Text with formatting
#        transform=ax.transAxes,  # Use axes coordinates
#        fontsize=24,
#        verticalalignment='top',
#        horizontalalignment='right',
#        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
#    )
    # Improve aesthetics
    plt.grid()
#    plt.grid(True, linestyle="--", alpha=0.6)  # Add a light dashed grid
    plt.tight_layout()

    # Save and show
    plt.savefig(f"./{PSR_name}_residuals_difference.pdf")
    plt.show()
    plt.close()