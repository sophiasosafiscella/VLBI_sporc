import sys
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import LinAlgError

import numpy as np
import pandas as pd

from pint.toa import get_TOAs
from pint.models import get_model
from pint.residuals import Residuals
import pint.fitter
from pint_pal import noise_utils

import astropy.units as u

from VLBI_utils import calculate_prior, replace_params
import glob
import sys

def calculate_post(PSR_name: str, timing_solution, timfile: str, parfile: str, astrometric_data_file, resume=False, plot=False):
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.5)

    print(f"Processing iteration {timing_solution.Index} of {PSR_name}")

    # Load the TOAs
    toas = get_TOAs(timfile, planets=True)

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Plot the original timing residuals
    if plot:
        # Calculate residuals. Don't forget to use actual timing residuals!
        residuals = Residuals(toas, eq_timing_model).time_resids.to(u.us).value
        xt = toas.get_mjds()
        errors = toas.get_errors().to(u.us).value

        plt.figure()
        plt.errorbar(xt, residuals, yerr=errors, fmt='o')
        plt.title(str(PSR_name) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
            round(np.std(residuals), 2)))
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\mu s$)")
        plt.tight_layout()
        plt.savefig("./figures/residuals/" + PSR_name + "_" + str(timing_solution.Index) + "_pre.png")
        plt.show()

    # Unfreeze the EFAC and EQUAD noise parameters
    #        unfreeze_noise(eq_timing_model)   # Thanks, Michael!

    # Replace the timing parameter values in the model with those from the new timing solution
    eq_timing_model = replace_params(eq_timing_model, timing_solution)

    # Perform initial fit
    initial_fit = pint.fitter.DownhillGLSFitter(toas, eq_timing_model)
    try:
        initial_fit.fit_toas(maxiter=5)
    except LinAlgError:
        print(f"LinAlgError at iteration {timing_solution.Index}")

    # Re-run noise
    print("Re-running noise")
    noise_utils.model_noise(eq_timing_model, toas, vary_red_noise=True, n_iter=int(5e4), using_wideband=False,
                            resume=resume, run_noise_analysis=True, base_op_dir=f"noisemodel_linear_sd/timing_solution_{timing_solution.Index}/")
    newmodel = noise_utils.add_noise_to_model(eq_timing_model, save_corner=False, base_dir=f"noisemodel_linear_sd/timing_solution_{timing_solution.Index}/")
    print("Done!")

    # Final fit
    final_fit = pint.fitter.DownhillGLSFitter(toas, newmodel)
    try:
        print("Fitting the new model")
        final_fit.fit_toas(maxiter=5)
        final_fit_resids = final_fit.resids
        print("Done!")

        # Calculate the posterior for this model and TOAs
        posterior = calculate_prior(eq_timing_model, astrometric_data_file, PSR_name) * final_fit_resids.lnlikelihood()

    except LinAlgError:
        print(f"LinAlgError at iteration {timing_solution.Index}")
        posterior = [[0.0]]

    # Output the results
#    res_df = pd.DataFrame({'PMRA': [timing_solution.PMRA], 'PMDEC': [timing_solution.PMDEC],
#                           'PX': [timing_solution.PX], 'posterior': [posterior[0][0]]})
#    res_df.to_pickle(posteriors_dir + "/" + str(timing_solution.Index) + "_posterior.pkl")

    # Let's plot the residuals and compare
    if plot:
        plt.figure()
        plt.errorbar(
            xt,
            final_fit_resids.time_resids.to(u.us).value,
            toas.get_errors().to(u.us).value,
            fmt='o',
        )
        plt.title("%s Post-Fit Timing Residuals" % PSR_name)
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\mu s$)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("./figures/residuals/" + PSR_name + "_" + str(timing_solution.Index) + "_post.png")
        plt.show()

    return posterior


if __name__ == "__main__":
    PSR_name, idx, PMRA, PMDEC, PX = sys.argv[1:]  # Timing solution index and parameters

    resume: bool = True

    timing_solution_dict = {"Index": idx, "PMRA": PMRA, "PMDEC": PMDEC, "PX": PX}
    for t in pd.DataFrame(timing_solution_dict, columns=list(timing_solution_dict.keys())[1:], index=[timing_solution_dict['Index']]).itertuples(index=True):
        timing_solution = t

    posteriors_dir: str = f"./results/timing_posteriors/{PSR_name}"
    astrometric_data_file: str = "./data/astrometric_values.csv"

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Calculate the posterior
    posterior = calculate_post(PSR_name, timing_solution, timfile, parfile, astrometric_data_file, resume)[0][0]

    # Save the timing solution with its posterior
    res_np = np.asarray([idx, PMRA, PMDEC, PX, posterior])
    np.save(posteriors_dir + "/" + str(idx) + "_posterior.npy", res_np)
