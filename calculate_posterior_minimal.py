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