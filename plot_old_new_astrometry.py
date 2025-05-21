import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pint.residuals import Residuals
from pint.models import get_model
from astropy.coordinates import Angle
import astropy.units as u
from scipy import signal
import glob
import sys

from pint.toa import get_TOAs

from VLBI_utils import replace_params

if __name__ == "__main__":

    PSR_name: str = "J2145-0750"
    line_width: int = 1.5
    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

    # Load the timing solution
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0).loc[PSR_name]

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)

    # Convert PX, PMRA, PMDEC to float
    result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # Find the solution with the highest posterior
    best_sol_idx = result_df['posterior'].idxmax()
    best_sol = result_df.loc[best_sol_idx]

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6, 7))
    sns.set(style="ticks", palette="colorblind", color_codes=True, font_scale=2.5)
    sns.despine(fig=fig)

    # RAJ
    timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
    ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")
    timing_RAJ_ms = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
    RAJ_timing_error_ms = Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0
    RAJ_new = (Angle(best_sol.RAJ, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0

    axs[0].errorbar(timing_RAJ_ms, list(["RAJ"]), xerr=RAJ_timing_error_ms, fmt='o', capsize=5, elinewidth=line_width, color='#E40303')
    axs[0].scatter(RAJ_new, list(["RAJ"]), marker='x', lw=3, s=80, color='#E40303')
    axs[0].set_xlabel("$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$")

    # DECJ
    timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
    ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 1)}s")
    timing_DECJ_ms = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
    DECJ_timing_error_ms = Angle(timing_astrometric_data['dec_te'], unit=u.degree).dms[2] * 1000.0
    DECJ_new = (Angle(best_sol.DECJ, unit=u.degree) - ref_DECJ).dms[2] * 1000.0

    axs[1].errorbar(timing_DECJ_ms, list(["DECJ"]), xerr=DECJ_timing_error_ms, fmt='o', capsize=5, elinewidth=line_width, color='#FF8C00')
    axs[1].scatter(DECJ_new, list(["DECJ"]), marker='x', lw=3, s=80, color='#FF8C00')
    axs[1].set_xlabel("$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$")

    # PX
    axs[2].errorbar(timing_astrometric_data['px_t'], list(["PX"]), xerr=timing_astrometric_data['px_te'], fmt='o', capsize=5, elinewidth=line_width, color='#B4D800')
    axs[2].scatter(best_sol.PX, list(["PX"]), marker='x', lw=3, s=80, color='#B4D800')
    axs[2].set_xlabel("$\Pi~[\mathrm{mas}]$")

    # PMRA
    axs[3].errorbar(timing_astrometric_data['pmra_t'], list(["PMRA"]), xerr=timing_astrometric_data['pmra_te'], fmt='o', capsize=5, elinewidth=line_width, color='#004DFF')
    axs[3].scatter(best_sol.PMRA, list(["PMRA"]), marker='x', lw=3, s=80, color='#004DFF')
    axs[3].set_xlabel("$\mu_\\alpha~[\mathrm{mas/yr}]$")

    # PMRA
    axs[4].errorbar(timing_astrometric_data['pmdec_t'], list(["PMDEC"]), xerr=timing_astrometric_data['pmdec_te'], fmt='o', capsize=5, elinewidth=line_width, color='#750787')
    axs[4].scatter(best_sol.PMDEC, list(["PMDEC"]), marker='x', lw=3, s=80, color='#750787')
    axs[4].set_xlabel("$\mu_\delta~[\mathrm{mas/yr}]$")


    plt.suptitle(PSR_name, fontsize=16, y=0.92)
    plt.tight_layout(h_pad=0.1)
    plt.savefig(f"./figures/{PSR_name}_old_vs_new_astrometry.pdf")
    plt.show()