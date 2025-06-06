import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from pint.toa import get_TOAs
from pint.models import get_model
from pint.residuals import Residuals
import pint.fitter
import seaborn as sns

import glob

sns.set_style("ticks")

if __name__ == "__main__":
    PSR_name: str = "J0030+0451"

    fig, ax = plt.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0})

    # Hide x-axis labels for all but the last subplot
    for i in range(2):
        ax[i].set_xticks([])

    fig.suptitle(PSR_name)
    titles = ['Original Residuals', 'New Residuals', 'Difference']  # Replace with your desired titles
    for i in range(3):
        ax[i].text(0.7, 0.95, titles[i], transform=ax[i].transAxes, ha='left', va='top')

    # Names of the original .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Names of the new .par file
    new_parfile: str = glob.glob(f"./results/new_fits/{PSR_name}/*par")[0]

    # Load the TOAs
    toas = get_TOAs(timfile, planets=True)

    # Load the original timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Plot the original timing residuals
    # Calculate residuals. Don't forget to use actual timing residuals!
    original_residuals = Residuals(toas, eq_timing_model).time_resids.to(u.us).value
    xt = toas.get_mjds()
    errors = toas.get_errors().to(u.us).value

    ax[0].scatter(xt, original_residuals)
#    plt.errorbar(xt, original_residuals, yerr=errors, fmt='o')
#    plt.title(str(PSR_name) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
#        round(np.std(original_residuals), 2)))
#    plt.xlabel("MJD")
    ax[0].set_ylabel("Residual [$\mathrm{\mu s}$]")
#    plt.tight_layout()
#    plt.show()

    # Load the new timing model and convert to equatorial coordinates
    new_ec_timing_model = get_model(new_parfile)  # Ecliptical coordiantes
    new_eq_timing_model = new_ec_timing_model.as_ICRS(epoch=new_ec_timing_model.POSEPOCH.value)

    # Plot the original timing residuals
    # Calculate residuals. Don't forget to use actual timing residuals!
    new_residuals = Residuals(toas, new_eq_timing_model).time_resids.to(u.us).value
#    xt = toas.get_mjds()
#    errors = toas.get_errors().to(u.us).value

#    plt.figure()
    ax[1].scatter(xt, new_residuals)
#    plt.errorbar(xt, new_residuals, yerr=errors, fmt='o')
#    plt.title(str(PSR_name) + " New Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
#        round(np.std(new_residuals), 2)))
#    plt.xlabel("MJD")
    ax[1].set_ylabel("Residual [$\mathrm{\mu s}$]")
#    plt.tight_layout()
#    plt.show()

    # Plot the differences between both sets of residuals
#    plt.figure()
    ax[2].scatter(xt, original_residuals-new_residuals)
 #   plt.errorbar(xt, original_residuals-new_residuals, yerr=errors, fmt='o')
#    plt.ylim([-25.0, 25.0])
#    plt.title(str(PSR_name) + " Differences in Timing Residuals")
    ax[2].set_xlabel("MJD")
    ax[2].set_ylabel("Residual ($\mu s$)")
    plt.tight_layout()
    plt.show()

