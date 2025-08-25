import pint
import sys

from astropy.time import Time
from pint.models import get_model
from pint.toa import get_TOAs

from VLBI_utils import epoch_scrunch
from uncertainties import unumpy
import astropy.units as u
import glob
import numpy as np

PSR_name: str = sys.argv[1]
posteriors_dir: str = f"./results/timing_posteriors_frame_tie/{PSR_name}"

# Names of the .tim and .par files
timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}_PINT*tim")[0]
parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}_PINT*par")[0]

# Load the timing model and convert to equatorial coordinates
ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
original_epoch = Time(ec_timing_model.POSEPOCH.value, format='mjd', scale='tdb')
eq_timing_model = ec_timing_model.as_ICRS(epoch=original_epoch)

# Load the TOAs
toas = get_TOAs(timfile, planets=True, ephem=eq_timing_model.EPHEM.value)

# Calculate the original NANOGrav residuals (with the original timing model)
fitter_object = pint.fitter.DownhillGLSFitter(toas, ec_timing_model)
avg_dict = fitter_object.resids.ecorr_average(use_noise_model=True)
res_avg = avg_dict['time_resids'].to(u.us).value
res_avg_errs = avg_dict['errors'].to(u.us).value
avg_mjds = avg_dict['mjds'].value

# Average the observations at different frequencies within each time window
ng15_epochs, ng15_avg_residuals, ng_15_avg_errors = epoch_scrunch(avg_mjds, data=res_avg, errors=res_avg_errs,
                                                                  weighted=True)

ng15_res = unumpy.uarray(ng15_avg_residuals, ng_15_avg_errors)

np.save(posteriors_dir + "/ng15_res.npy", ng15_res)
