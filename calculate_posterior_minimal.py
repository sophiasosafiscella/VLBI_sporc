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


def calculate_post(PSR_name: str, timing_solution, timfile: str, parfile: str, astrometric_data_file, plot=False):
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.5)

    print(f"Processing iteration {timing_solution.Index} of {PSR_name}")

    # Load the TOAs
    toas = get_TOAs(timfile, planets=True)

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    print("Successfully converted from ecliptical to equatorial")
    return

if __name__ == "__main__":
    PSR_name, idx, PMRA, PMDEC, PX = sys.argv[1:]  # Timing solution index and parameters

    import astropy
    import pint

    print(astropy.__version__)
    print(pint.__version__)

    timing_solution_dict = {"Index": idx, "PMRA": PMRA, "PMDEC": PMDEC, "PX": PX}
    for t in pd.DataFrame(timing_solution_dict, columns=list(timing_solution_dict.keys())[1:], index=[timing_solution_dict['Index']]).itertuples(index=True):
        timing_solution = t

    posteriors_dir: str = f"./results/timing_posteriors/{PSR_name}"
    astrometric_data_file: str = "./data/astrometric_values.csv"

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Calculate the posterior
    posterior = calculate_post(PSR_name, timing_solution, timfile, parfile, astrometric_data_file)

    # Save the timing solution with its posterior
#    res_np = np.asarray([idx, PMRA, PMDEC, PX, posterior])
#    np.save(posteriors_dir + "/" + str(idx) + "_posterior.npy", res_np)