import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import plotly.colors
from astropy.coordinates import Angle, SkyCoord, ICRS
from astropy.time import Time
import astropy.units as u

from VLBI_utils import pdf_values, Wang_frame_tie
from uncertainties import ufloat, umath
import sys


VLBI_data = pd.read_csv("./data/calibrated_vlbi_astrometric_data.csv", header=0, index_col=0)
timing_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", header=0, index_col=0)
PSR_list = VLBI_data.index
sns.set(context="paper", style="darkgrid", font_scale=1.9)
sns.set_palette(plotly.colors.qualitative.Plotly)
# Set Plotly-style background color
plotly_bg = "#e5ecf6"

fig, axs = plt.subplots(nrows=8, ncols=6, figsize=(18,26), constrained_layout=True)

VLBI_color = (0/255, 204/255, 150/255, 0.5)   # teal-like green with alpha
timing_color = (99/255, 110/255, 250/255, 0.5) # bluish with alpha

# Frame tie
A = pd.read_csv('./data/NG_frame_tie/NG_frame_tie.csv', header=0).to_dict('records')[0]

# Make sure Omega is in radians and not mas
Ax, Ay, Az = Angle(A['Ax'], u.mas).rad, Angle(A['Ay'], u.mas).rad, Angle(A['Az'], u.mas).rad
Omega = np.array([[1.0, Az, -1.0 * Ay], [-1.0 * Az, 1.0, Ax], [Ay, -1.0 * Ax, 1.0]])

for i, PSR in enumerate(PSR_list[9:]):

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for position
    # ----------------------------------------------------------------------------------------------

    VLBI_pos_ICRF = SkyCoord(ra=VLBI_data.loc[PSR, "ra_v"], dec=VLBI_data.loc[PSR, "dec_v"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    VLBI_pos_ICRF_err = SkyCoord(ra=VLBI_data.loc[PSR, "ra_ve"], dec=VLBI_data.loc[PSR, "dec_ve"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    # Create uncertainty objects to handle error propagation
    VLBI_pos_ICRF_spherical = dict(ra=ufloat(VLBI_pos_ICRF.ra.rad, VLBI_pos_ICRF_err.ra.rad),
                                dec=ufloat(VLBI_pos_ICRF.dec.rad, VLBI_pos_ICRF_err.dec.rad))

    # Apply the frame tie
    VLBI_pos_SBB_spherical = Wang_frame_tie(VLBI_pos_ICRF_spherical, Omega, astropy=True)

    # Create SkyCoord objects for the positions in the SSB system
    VLBI_pos_SSB = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].nominal_value, dec=VLBI_pos_SBB_spherical["dec"].nominal_value,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    VLBI_pos_SSB_err = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].std_dev, dec=VLBI_pos_SBB_spherical["dec"].std_dev,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    # Output the corrected positions
    VLBI_data.loc[PSR, "ra_v"] = Angle(VLBI_pos_SSB.ra).to_string(unit=u.hourangle)
    VLBI_data.loc[PSR, "dec_v"] = VLBI_pos_SSB.dec.to_string(unit=u.degree)
    VLBI_data.loc[PSR, "ra_ve"] = VLBI_pos_SSB_err.ra.to_string(unit=u.hourangle)
    VLBI_data.loc[PSR, "dec_ve"] = VLBI_pos_SSB_err.dec.to_string(unit=u.degree)

#   VLBI_data.to_csv("./data/frame_tied_vlbi_astrometric_data.csv")

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for proper motion
    # ----------------------------------------------------------------------------------------------
    for error_side in ["uL", "uR"]:
        VLBI_PM_ICRF = dict(PMRA=Angle(VLBI_data.loc[PSR, "pmra_v"], unit=u.mas),
                            PMDEC=Angle(VLBI_data.loc[PSR, "pmdec_v"], unit=u.mas))

        VLBI_PM_ICRF_err = dict(PMRA=Angle(VLBI_data.loc[PSR, f"pmra_v_{error_side}"], unit=u.mas),
                                PMDEC=Angle(VLBI_data.loc[PSR, f"pmdec_v_{error_side}"], unit=u.mas))

        # Create uncertainty objects to handle error propagation
        VLBI_PM_ICRF_spherical = dict(ra=ufloat(VLBI_PM_ICRF['PMRA'].rad, VLBI_PM_ICRF_err['PMRA'].rad),
                                      dec=ufloat(VLBI_PM_ICRF['PMDEC'].rad, VLBI_PM_ICRF_err['PMDEC'].rad))

        # Apply the frame tie
        VLBI_PM_SBB_spherical = Wang_frame_tie(VLBI_PM_ICRF_spherical, Omega = np.identity(3), astropy=False)

        if error_side == "uL":
            VLBI_PMRA_SSB_uL = VLBI_PM_SBB_spherical["ra"].std_dev
            VLBI_PMDEC_SSB_uL = VLBI_PM_SBB_spherical["dec"].std_dev
        elif error_side == "uR":
            VLBI_PMRA_SSB_uR = VLBI_PM_SBB_spherical["ra"].std_dev
            VLBI_PMDEC_SSB_uR = VLBI_PM_SBB_spherical["dec"].std_dev

    VLBI_PM_SSB = dict(PMRA=Angle(VLBI_PM_SBB_spherical["ra"].nominal_value, unit=u.rad).to(u.mas).value,
                        PMDEC=Angle(VLBI_PM_SBB_spherical["dec"].nominal_value, unit=u.rad).to(u.mas).value)

    VLBI_PM_SSB_err = dict(PMRA_uL=Angle(VLBI_PMRA_SSB_uL, unit=u.rad).to(u.mas).value,
                           PMRA_uR=Angle(VLBI_PMRA_SSB_uR, unit=u.rad).to(u.mas).value,
                           PMDEC_uL=Angle(VLBI_PMDEC_SSB_uL, unit=u.rad).to(u.mas).value,
                           PMDEC_uR=Angle(VLBI_PMDEC_SSB_uR, unit=u.rad).to(u.mas).value)

    #----------------------------------------------------------------------------------------------
    # Timing positions
    # ----------------------------------------------------------------------------------------------
    timing_pos_SSB = SkyCoord(ra=timing_data.loc[PSR, "ra_t"], dec=timing_data.loc[PSR, "dec_t"],
                              frame=ICRS, unit=(u.hourangle, u.deg),
                              equinox=timing_data.loc[PSR, "equinox"],
                              obstime=Time(val=timing_data.loc[PSR, "epoch_t"], format='mjd', scale='utc'))

    timing_pos_SSB_err = SkyCoord(ra=timing_data.loc[PSR, "ra_te"], dec=timing_data.loc[PSR, "dec_te"],
                                  frame=ICRS, unit=(u.hourangle, u.deg),
                                  equinox=timing_data.loc[PSR, "equinox"],
                                  obstime=Time(val=timing_data.loc[PSR, "epoch_t"], format='mjd', scale='utc'))

    # ------------------------------RAJ------------------------------
    ref_RAJ = Angle(f"{int(VLBI_pos_SSB.ra.hms[0])}h{int(VLBI_pos_SSB.ra.hms[1])}m{round(VLBI_pos_SSB.ra.hms[2], 1)}s")

    # VLBI
    VLBI_deltaRAJ_ms = (VLBI_pos_SSB.ra.to(u.hourangle) - ref_RAJ).hms[2] * 1000.0
    VLBI_deltaRAJ_err_ms = VLBI_pos_SSB_err.ra.hms[2] * 1000.0
    x_VLBI_RAJ, y_VLBI_RAJ = pdf_values(x0=VLBI_deltaRAJ_ms, uL=VLBI_deltaRAJ_err_ms, uR=VLBI_deltaRAJ_err_ms)

    axs[i, 0].fill_between(x_VLBI_RAJ, y_VLBI_RAJ, color=VLBI_color, linewidth=0, label=None)

    # Timing
    timing_deltaRAJ_ms = (timing_pos_SSB.ra.to(u.hourangle) - ref_RAJ).hms[2] * 1000.0
    timing_RAJ_err_ms = timing_pos_SSB_err.ra.hms[2] * 1000.0
    x_timing_RAJ, y_timing_RAJ = pdf_values(x0=timing_deltaRAJ_ms, uL=timing_RAJ_err_ms, uR=timing_RAJ_err_ms)
    x_timing_RAJ, y_timing_RAJ = [np.float64(z) for z in x_timing_RAJ], [np.float64(z) for z in y_timing_RAJ]

    axs[i, 0].fill_between(x_timing_RAJ, y_timing_RAJ, color=timing_color, linewidth=0, label=None)
    axs[i, 0].set_xlabel("$\\alpha -$ " + f"${ref_RAJ:latex}$"[1:-1] + "\n" + "$[\mathrm{mas}]$")

 # ------------------------------DECJ------------------------------
    timing_DECJ = ufloat(Angle(timing_data.loc[PSR, 'dec_t']).rad, Angle(timing_data.loc[PSR, "dec_te"]).rad)
    VLBI_DECJ = ufloat(Angle(VLBI_data.loc[PSR, "dec_v"]).rad, Angle(VLBI_data.loc[PSR, "dec_ve"]).rad)

    ref_DECJ = Angle(f"{int(VLBI_pos_SSB.dec.dms[0])}d{int(abs(VLBI_pos_SSB.dec.dms[1]))}m{int(abs(VLBI_pos_SSB.dec.dms[2]))}s")

    # VLBI
    VLBI_deltaDECJ_ms = (VLBI_pos_SSB.dec.to(u.degree) - ref_DECJ).dms[2] * 1000.0
    VLBI_DECJ_err_ms = VLBI_pos_SSB_err.dec.dms[2] * 1000.0

    x_VLBI_DECJ, y_VLBI_DECJ = pdf_values(x0=VLBI_deltaDECJ_ms, uL=VLBI_DECJ_err_ms, uR=VLBI_DECJ_err_ms)
    axs[i, 1].fill_between(x_VLBI_DECJ, y_VLBI_DECJ, color=VLBI_color, linewidth=0, label=None)

    # Timing
    timing_deltaDECJ_ms = (timing_pos_SSB.dec.to(u.degree) - ref_DECJ).dms[2] * 1000.0
    timing_DECJ_err_ms = timing_pos_SSB_err.dec.dms[2] * 1000.0
    x_timing_DEJC, y_timing_DECJ = pdf_values(x0=timing_deltaDECJ_ms, uL=timing_DECJ_err_ms, uR=timing_DECJ_err_ms)

    axs[i, 1].fill_between(x_timing_DEJC, y_timing_DECJ, color=timing_color, linewidth=0, label=None)
    axs[i, 1].set_xlabel("$\delta -$ (" + f"${ref_DECJ:latex}$"[1:-1] + ")" + "\n" + "$[\mathrm{mas}]$")

    # ------------------------------Parallax------------------------------
    # VLBI
    x_VLBI_PX, y_VLBI_PX = pdf_values(x0=VLBI_data.loc[PSR, "px_v"], uL=VLBI_data.loc[PSR, "px_v_uL"], uR=VLBI_data.loc[PSR, "px_v_uR"])
    axs[i, 2].fill_between(x_VLBI_PX, y_VLBI_PX, color=VLBI_color, linewidth=0, label=None)

    # Timing
    x_timing_PX, y_timing_PX = pdf_values(x0=timing_data.loc[PSR, "px_t"], uL=timing_data.loc[PSR, "px_te"],
                      uR=timing_data.loc[PSR, "px_te"])

    axs[i, 2].fill_between(x_timing_PX, y_timing_PX, color=timing_color, linewidth=0, label=None)
    axs[i, 2].set_xlabel("$\Pi [\mathrm{mas}]$")

    # ------------------------------Proper Motion------------------------------
    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(VLBI_PM_SSB['PMRA'], VLBI_PM_SSB_err['PMRA_' + error_side])
        VLBI_PMDEC = ufloat(VLBI_PM_SSB['PMDEC'], VLBI_PM_SSB_err['PMDEC_' + error_side])

        # As far as I can tell, all the papers where I extracted the values of PMRA already include the cos(delta) in
        # the definition for PMRA. That is, PMRA = dalpha/dt * cos(delta)
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2)

        if error_side == "uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    x_VLBI_PM, y_VLBI_PM = pdf_values(x0=VLBI_PM.nominal_value, uL=VLBI_PM_uL, uR=VLBI_PM_uR)
    axs[i, 3].fill_between(x_VLBI_PM, y_VLBI_PM, color=VLBI_color, linewidth=0, label=None)

    # Timing
    timing_PMRA = ufloat(timing_data.loc[PSR, "pmra_t"], timing_data.loc[PSR, "pmra_te"])
    timing_PMDEC = ufloat(timing_data.loc[PSR, "pmdec_t"], timing_data.loc[PSR, "pmdec_te"])

    # As far as I can tell, all the papers where I extracted the values of PMRA already include the cos(delta) in
    # the definition for PMRA. That is, PMRA = dalpha/dt * cos(delta)
    timing_PM = umath.sqrt(timing_PMDEC ** 2 + timing_PMRA ** 2)

    x_timing_PM, y_timing_PM = pdf_values(x0=timing_PM.nominal_value, uL=timing_PM.std_dev, uR=timing_PM.std_dev)
    axs[i, 3].fill_between(x_timing_PM, y_timing_PM, color=timing_color, linewidth=0, label=None)
    axs[i, 3].set_xlabel("$\mu~[\mathrm{mas~yr^{-1}}]$")

    # ------------------------------PMRA------------------------------
    # VLBI
    x_VLBI_PMRA, y_VLBI_PMRA = pdf_values(x0=VLBI_PM_SSB['PMRA'], uL=VLBI_PM_SSB_err['PMRA_uL'], uR=VLBI_PM_SSB_err['PMRA_uR'])
    axs[i, 4].fill_between(x_VLBI_PMRA, y_VLBI_PMRA, color=VLBI_color, linewidth=0, label=None)

    # Timing
    x_timing_PMRA, y_timing_PMRA = pdf_values(x0=timing_PMRA.nominal_value, uL=timing_PMRA.std_dev, uR=timing_PMRA.std_dev)
    axs[i, 4].fill_between(x_timing_PMRA, y_timing_PMRA, color=timing_color, linewidth=0, label=None)
    axs[i, 4].set_xlabel("$\mu_{\mathrm{RA}}~[\mathrm{mas~yr^{-1}}]$")
    # ------------------------------PMDEC------------------------------
    # VLBI
    x_VLBI_PMDEC, y_VLBI_PMDEC = pdf_values(x0=VLBI_PM_SSB['PMDEC'], uL=VLBI_PM_SSB_err['PMDEC_uL'], uR=VLBI_PM_SSB_err['PMDEC_uR'])
    axs[i, 5].fill_between(x_VLBI_PMDEC, y_VLBI_PMDEC, color=VLBI_color, linewidth=0, label=None)

    # Timing
    x_timing_PMDEC, y_timing_PMDEC = pdf_values(x0=timing_PMDEC.nominal_value, uL=timing_PMDEC.std_dev, uR=timing_PMDEC.std_dev)
    axs[i, 5].fill_between(x_timing_PMDEC, y_timing_PMDEC, color=timing_color, linewidth=0, label=None)
    axs[i, 5].set_xlabel("$\mu_{\mathrm{DEC}}~[\mathrm{mas~yr^{-1}}]$")

    axs[i, 0].set_ylabel(PSR)

    for j in range(6):
        axs[i, j].set_facecolor(plotly_bg)
        axs[i, j].xaxis.set_major_locator(MaxNLocator(prune='both', nbins='auto'))

#fig.tight_layout()
fig.savefig(f"./figures/PDFs_with_frametie_matplotlib_2.pdf")
fig.show()