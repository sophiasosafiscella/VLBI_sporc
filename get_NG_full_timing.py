import numpy as np
import pandas as pd
import contextlib
import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import glob
import sys

VLBI_data = pd.read_csv('./data/NG_frame_tie/NG_msp_vlbi.csv', sep=',', comment='#', index_col=0)
PSR_list = VLBI_data.index.tolist()
n_psr = len(PSR_list)
new_epochs = VLBI_data['epoch_v'].to_numpy()

RA_list, RA_err_list, DEC_list, DEC_err_list, EPHEM_list = [np.empty(n_psr, dtype=object) for _ in range(5)]
PMRA_list, PMRA_err_list, PMDEC_list, PMDEC_err_list, PX_list, PX_err_list = [np.empty(n_psr, dtype=float) for _ in range(6)]
POSEPOCH_list = np.empty(n_psr, dtype=float)

for k, psr in enumerate(PSR_list):

    PSR_name = psr + "_PINT"
    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)                                # Ecliptical coordiantes
    eq = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # Update the epoch to match that of VLBI
    eq.change_posepoch(new_epochs[k])

    RA_list[k] = eq.RAJ.quantity
    RA_err_list[k] = eq.RAJ.uncertainty
    DEC_list[k] = eq.DECJ.quantity
    DEC_err_list[k] = eq.DECJ.uncertainty

    PMRA_list[k] = eq.PMRA.value
    PMRA_err_list[k] = eq.PMRA.uncertainty.value
    PMDEC_list[k] = eq.PMDEC.value
    PMDEC_err_list[k] = eq.PMDEC.uncertainty.value

    PX_list[k] = eq.PX.value
    PX_err_list[k] = eq.PX.uncertainty.value

    POSEPOCH_list[k] = eq.POSEPOCH.value
    EPHEM_list[k] = eq.EPHEM.value


data = pd.DataFrame({'epoch_t': POSEPOCH_list, 'ephem': EPHEM_list,
                     't_RAJ': RA_list, 't_RAJ_err': RA_err_list, 't_DECJ': DEC_list, 't_DECJ_err': DEC_err_list,
                     't_PMRA': PMRA_list, 't_PMRA_err': PMRA_err_list, 't_PMDEC': PMDEC_list, 't_PMDEC_err': PMDEC_err_list,
                     't_PX': PX_list, 't_PX_err': PX_err_list}, index=PSR_list)

data.to_csv("./data/timing_astrometric_data.csv")

