# Calculate residuals after replacing the astrometric parameters in the timing model with the maximum posterior
# astrometric parameters derived from using VLBI priors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pint.residuals import Residuals
from pint.models import get_model
import astropy.units as u
from astropy.timeseries import LombScargle
from scipy import signal

from VLBI_utils import epoch_scrunch

import glob
import os
import sys

from pint.toa import get_TOAs

from VLBI_utils import replace_params

if __name__ == "__main__":

#    PSR_name: str = "J0030+0451"
#    PSR_name: str = "J2145-0750"
    PSR_name: str = "J1640+2224"
    file: str = f"./results/{PSR_name}_new_res_diff.csv"

    if os.path.isfile(file):
        data = pd.read_csv(file)
        xt = data['MJD']
        res_diff = data['Residuals Difference']
        errors = data['Error']
        median_error = np.median(errors)

    else:
        posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

        # Names of the .tim and .par files
        timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}_PINT*tim")[0]
        parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}_PINT*par")[0]

        # Get the nominal timing values
        ec_timing_model = get_model(parfile)                                             # Ecliptical coordiantes
        eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

        # Get the TOAs
        toas = get_TOAs(timfile, planets=True, ephem=eq_timing_model.EPHEM.value)

        # Load the posteriors
        result_df = pd.read_pickle(posteriors_file)

        # Convert PX, PMRA, PMDEC to float
        result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

        # Find the solution with the highest posterior
        best_sol_idx = result_df['posterior'].idxmax()
        best_sol = result_df.loc[best_sol_idx]

        # Calculate the original residuals
        xt = toas.get_mjds().value
        errors = toas.get_errors().value
        median_error = np.median(errors)
        rs = Residuals(toas, eq_timing_model).resids.to(u.us).value

        # Epoch-average
#        epoch_average_rs = epoch_scrunch(toas, data=rs)

        # Replace the old astrometric values with the maximum posterior ones
        replace_params(eq_timing_model, best_sol)

        # Calculate the new residuals
        new_rs = Residuals(toas, eq_timing_model).resids.to(u.us).value
#        new_epoch_average_rs = epoch_scrunch(toas, data=new_rs)

        res_diff = rs - new_rs
#        res_diff = epoch_average_rs - new_epoch_average_rs

        data = pd.DataFrame({'MJD': xt, 'Residuals Difference': res_diff, 'Error': errors})
        data.to_csv(file, index=False)

    epochs, avg_residuals, avg_errors = epoch_scrunch(xt, data=res_diff, errors=errors, weighted=True)

    # Now create the figure
    sns.set(context="paper", style="ticks", font_scale=3.0)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(epochs, avg_residuals)

    ax.set_title(f"{PSR_name} Difference in Pre-Fit Timing Residuals")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residuals Difference ($\mu s$)")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.98, 0.95,  # X and Y position in *axes coordinates* (0 to 1)
        f"Error Median = {median_error:.2f}",  # Text with formatting
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=24,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    # Improve aesthetics
    plt.grid()
#    plt.grid(True, linestyle="--", alpha=0.6)  # Add a light dashed grid
    plt.tight_layout()

    # Save and show
    plt.savefig(f"./figures/{PSR_name}_residuals_difference.pdf")
    plt.show()
    plt.close()