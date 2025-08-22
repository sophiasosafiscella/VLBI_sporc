import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u
from astropy.timeseries import LombScargle

from VLBI_utils import epoch_scrunch

import os

if __name__ == "__main__":

    PSR_name: str = "J0030+0451"
#    PSR_name: str = "J2145-0750"
#    PSR_name: str = "J1640+2224"

    data = pd.read_csv(f"./{PSR_name}_res_diff.csv")
    xt = data['MJD']
    res_diff = data['Residuals Difference']
    errors = data['Error']
    median_error = np.median(errors)

    epochs = xt
    avg_residuals = res_diff
    avg_errors = errors

    # Lomb-Scargle Periodogram
    frequency, power = LombScargle(epochs * u.day, avg_residuals * u.us).autopower()
    sns.set(context="paper", style="ticks", font_scale=3.0)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14)) #, gridspec_kw = {'wspace':0, 'hspace':0})
#    plt.suptitle(PSR_name.replace("-", "$-$"))

    axs[0].errorbar(epochs, avg_residuals, yerr=avg_errors/10.0, fmt='o', capsize=5, capthick=0.5)
#    axs[0].scatter(epochs, avg_residuals)
#    axs[0].set_ylim([-0.75,1.0])
    axs[0].set_xlabel("Observing epoch (MJD)")
    axs[0].set_ylabel("Residuals difference [$\mathrm{\mu s}$] \n (Re-scaled uncertainties)")
    axs[0].grid()

    '''
    axs[0].plot(frequency[1:], power[1:])
    axs[0].axvline(x=1.0/((1 * u.yr).to(u.day)).value, color='C1', linestyle='--', lw=2.0, label="$(\mathrm{1~year})^{-1}$")
    axs[0].axvline(x=1.0/((0.5 * u.yr).to(u.day)).value, color='C2', linestyle='--', lw=2.0, label="$(\mathrm{6~months})^{-1}$")
    axs[0].set_xscale('log')
    axs[0].set_xticklabels([])
#    ax.set_yscale('log')
    axs[0].set_ylabel('Power [$\mathrm{\mu s^2}$/day]')
    axs[0].grid()
    axs[0].legend(loc='upper left')
    '''

    axs[1].plot(frequency[1:], power[1:])
    axs[1].axvline(x=1.0/((1 * u.yr).to(u.day)).value, color='C1', linestyle='--', lw=2.0, label="$(\mathrm{1~year})^{-1}$")
    axs[1].axvline(x=1.0/((0.5 * u.yr).to(u.day)).value, color='C2', linestyle='--', lw=2.0, label="$(\mathrm{6~months})^{-1}$")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Frequency [1/day]')
    axs[1].set_ylabel('Log(Power) [$\mathrm{\mu s^2}$/day]')
    axs[1].legend(loc='best')
    axs[1].grid()


    plt.tight_layout()
    plt.savefig(f"./{PSR_name}_periodogram_paper.pdf")
    plt.show()