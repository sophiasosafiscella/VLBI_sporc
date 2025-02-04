import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pint import models

from astropy.coordinates import Angle, SkyCoord, ICRS, GCRS
from astropy.time import Time
import astropy.units as u
from VLBI_utils import pdf_values, Wang_frame_tie
from astropy.coordinates import solar_system_ephemeris

from uncertainties import ufloat, umath

import glob
import sys

VLBI_data = pd.read_csv("./data/VLBI_astrometric_values.csv", index_col=0)
timing_data = pd.read_csv("./data/timing_astrometric_data.csv", index_col=0)
timing_source: str = "NG"
PSR_list = VLBI_data.index
fig = make_subplots(rows=len(PSR_list), cols=6, vertical_spacing=0.02)
VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

# Frame tie
A_ufloat = [ufloat(Angle(0.57, unit=u.mas).rad, Angle(0.34, unit=u.mas).rad),
            ufloat(Angle(0.40, unit=u.mas).rad, Angle(0.81, unit=u.mas).rad),
            ufloat(Angle(-0.44, unit=u.mas).rad, Angle(0.73, unit=u.mas).rad)]
A = [A_ufloat[0].nominal_value, A_ufloat[1].nominal_value, A_ufloat[2].nominal_value]
Omega = np.array([[1.0, A[2], -1.0 * A[1]], [-1.0 * A[2], 1.0, A[0]], [A[1], -1.0 * A[0], 1.0]])

for i, PSR in enumerate(PSR_list):

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for position
    # ----------------------------------------------------------------------------------------------
    VLBI_pos_ICRF = SkyCoord(ra=VLBI_data.loc[PSR, "VLBI_RAJ"], dec=VLBI_data.loc[PSR, "VLBI_DECJ"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

    VLBI_pos_ICRF_err = SkyCoord(ra=VLBI_data.loc[PSR, "VLBI_RAJ_err"], dec=VLBI_data.loc[PSR, "VLBI_DECJ_err"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

    # Create uncertainty objects to handle error propagation
    VLBI_pos_ICRF_spherical = dict(ra=ufloat(VLBI_pos_ICRF.ra.rad, VLBI_pos_ICRF_err.ra.rad),
                                dec=ufloat(VLBI_pos_ICRF.dec.rad, VLBI_pos_ICRF_err.dec.rad))

    VLBI_pos_SBB_spherical = Wang_frame_tie(VLBI_pos_ICRF_spherical, Omega)

    with solar_system_ephemeris.set('de435'):
        VLBI_pos_SSB = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].nominal_value, dec=VLBI_pos_SBB_spherical["dec"].nominal_value,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

    print(VLBI_pos_SSB.ra.rad)
    print(VLBI_pos_SSB.dec.rad)

    with solar_system_ephemeris.set('de440'):
        VLBI_pos_SSB.transform_to(ICRS())

    print(VLBI_pos_SSB.ra.rad)
    print(VLBI_pos_SSB.dec.rad)
    sys.exit()

    VLBI_pos_SSB_err = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].std_dev, dec=VLBI_pos_SBB_spherical["dec"].std_dev,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for proper motion
    # ----------------------------------------------------------------------------------------------
    '''
    for error_side in ["uL", "uR"]:
        VLBI_PM_ICRF = dict(PMRA=Angle(VLBI_data.loc[PSR, "VLBI_PMRA"], unit=u.mas),
                            PMDEC=Angle(VLBI_data.loc[PSR, "VLBI_PMDEC"], unit=u.mas))

        VLBI_PM_ICRF_err = dict(PMRA=Angle(VLBI_data.loc[PSR, f"VLBI_PMRA_{error_side}"], unit=u.mas),
                                PMDEC=Angle(VLBI_data.loc[PSR, f"VLBI_PMDEC_{error_side}"], unit=u.mas))

        # Create uncertainty objects to handle error propagation
        VLBI_PM_ICRF_ufloat = np.array([ufloat(VLBI_PM_ICRF["PMRA"].rad, VLBI_PM_ICRF_err["PMRA"].rad),
                                  ufloat(VLBI_PM_ICRF["PMDEC"].rad, VLBI_PM_ICRF_err["PMDEC"].rad)])

        # Transform the (RA,DEC) to cartesian components in the ICRF. Do the error propagation automatically
        VLBI_PM_ICRF_xyz = umath_spherical_to_cartesian(VLBI_PM_ICRF_ufloat)
        PM_x, PM_y, PM_z = spherical_to_cartesian(1.0, VLBI_PM_ICRF["PMDEC"].rad, VLBI_PM_ICRF["PMRA"].rad)

        # Transform to xyz coordinates from the ICRF to the SSB framex
#        Omega = np.matrix(np.identity(3))
        VLBI_PM_SSB_xyz = np.array(np.dot(Omega, VLBI_PM_ICRF_xyz))[0]
        SSB_PM_x, SSB_PM_y, SSB_PM_z = np.array(np.dot(Omega, np.array([PM_x, PM_y, PM_z])))[0]

        # Transform cartesian components in the SSB frame to (RA,DEC)
        VLBI_PMRA_SSB, VLBI_PMDEC_SSB = umath_cartesian_to_spherical(VLBI_PM_SSB_xyz)
        r, pmdec, pmra = cartesian_to_spherical(SSB_PM_x, SSB_PM_y, SSB_PM_z)
        pmra = pmra - 2.0 * np.pi * u.rad

#        print(VLBI_PMRA_SSB.nominal_value, VLBI_PMDEC_SSB.nominal_value)
#        print(r, pmra, pmdec)
#        print(" ")

#        print(Angle(VLBI_PMRA_SSB.nominal_value, unit=u.rad).mas, Angle(VLBI_PMDEC_SSB.nominal_value, unit=u.rad).mas)
#        print(Angle(pmra, unit=u.rad).mas, Angle(pmdec, unit=u.rad).mas)
#        print(" ")

        if error_side == "uL":
            VLBI_PMRA_SSB_uL = VLBI_PMRA_SSB.std_dev
            VLBI_PMDEC_SSB_uL = VLBI_PMDEC_SSB.std_dev
        elif error_side == "uR":
            VLBI_PMRA_SSB_uR = VLBI_PMRA_SSB.std_dev
            VLBI_PMDEC_SSB_uR = VLBI_PMDEC_SSB.std_dev

    VLBI_PM_SSB = dict(PMRA=Angle(VLBI_PMRA_SSB.nominal_value, unit=u.rad).to(u.mas),
                        PMDEC=Angle(VLBI_PMDEC_SSB.nominal_value, unit=u.rad).to(u.mas))

    VLBI_PM_SSB_err = dict(PMRA_uL=Angle(VLBI_PMRA_SSB_uL, unit=u.rad).to(u.mas),
                           PMRA_uR=Angle(VLBI_PMRA_SSB_uR, unit=u.rad).to(u.mas),
                           PMRDEC_uL=Angle(VLBI_PMDEC_SSB_uL, unit=u.rad).to(u.mas),
                           PMRDEC_uR=Angle(VLBI_PMDEC_SSB_uR, unit=u.rad).to(u.mas))
    '''

    #----------------------------------------------------------------------------------------------
    # Equatorial timing model
    # ----------------------------------------------------------------------------------------------
    ec_timing_model = models.get_model(glob.glob(f"./data/NG_15yr_dataset/par/{PSR}*.nb.par")[0])   # Ecliptical coordiantes
#    eq_timing_model = ec_timing_model.as_ICRS()  # Equatorial coordinates
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # ------------------------------RAJ------------------------------
    ref_RAJ = Angle(f"{int(VLBI_pos_SSB.ra.hms[0])}h{int(VLBI_pos_SSB.ra.hms[1])}m{round(VLBI_pos_SSB.ra.hms[2], 1)}s")

    VLBI_deltaRAJ_ms = (VLBI_pos_SSB.ra.to(u.hourangle) - ref_RAJ).hms[2] * 1000.0
    VLBI_deltaRAJ_err_ms = VLBI_pos_SSB_err.ra.hms[2] * 1000.0

    x_VLBI_RAJ, y_VLBI_RAJ = pdf_values(x0=VLBI_deltaRAJ_ms, uL=VLBI_deltaRAJ_err_ms, uR=VLBI_deltaRAJ_err_ms)

    if i==0:
        fig.add_trace(go.Scatter(x=x_VLBI_RAJ, y=y_VLBI_RAJ, name="VLBI", fill='tozeroy', fillcolor=VLBI_color, mode='none'), row=i+1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x_VLBI_RAJ, y=y_VLBI_RAJ, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=1)

    # timing
    if timing_source == "NG":
        timing_deltaRAJ_ms = (Angle(eq_timing_model.RAJ.quantity) - ref_RAJ).hms[2] * 1000.0
        timing_RAJ_err_ms = Angle(eq_timing_model.RAJ.uncertainty).hms[2] * 1000.0
    elif timing_source == "PPTA":
        timing_pos_SSB = SkyCoord(ra=timing_data.loc[PSR, "timing_RAJ"], dec=timing_data.loc[PSR, "timing_DECJ"],
                                 frame=ICRS, unit=(u.hourangle, u.deg),
                                 equinox=timing_data.loc[PSR, "equinox"],
                                 obstime=Time(val=timing_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

        timing_pos_SSB_err = SkyCoord(ra=timing_data.loc[PSR, "timing_RAJ_err"], dec=timing_data.loc[PSR, "timing_DECJ_err"],
                                     frame=ICRS, unit=(u.hourangle, u.deg),
                                     equinox=timing_data.loc[PSR, "equinox"],
                                     obstime=Time(val=timing_data.loc[PSR, "POSEPOCH"], format='mjd', scale='utc'))

        timing_deltaRAJ_ms = (timing_pos_SSB.ra.to(u.hourangle) - ref_RAJ).hms[2] * 1000.0
        timing_RAJ_err_ms = timing_pos_SSB_err.ra.hms[2] * 1000.0

#    if PSR == "J0437-4715":
#        timing_deltaRAJ_ms = (Angle("04h37m15.883185s") - ref_RAJ).hms[2] * 1000.0
#        timing_RAJ_err_ms = Angle("0h0m0.000006s").hms[2] * 1000.0

    x_timing_RAJ, y_timing_RAJ = pdf_values(x0=timing_deltaRAJ_ms, uL=timing_RAJ_err_ms, uR=timing_RAJ_err_ms)
    x_timing_RAJ = [np.float64(z) for z in x_timing_RAJ]
    y_timing_RAJ = [np.float64(z) for z in y_timing_RAJ]
#    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=1)

    if i==0:
        fig.add_trace(go.Scatter(x=x_timing_RAJ, y=y_timing_RAJ, name="Timing", fill='tozeroy', fillcolor=timing_color, mode='none'), row=i+1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x_timing_RAJ, y=y_timing_RAJ, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=1)

    fig.update_xaxes(title_text="$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$", row=i+1, col=1)

    # ------------------------------DECJ------------------------------
    ref_DECJ = Angle(f"{int(VLBI_pos_SSB.dec.dms[0])}d{int(abs(VLBI_pos_SSB.dec.dms[1]))}m{int(abs(VLBI_pos_SSB.dec.dms[2]))}s")

    # VLBI
    VLBI_deltaDECJ_ms = (VLBI_pos_SSB.dec.to(u.degree) - ref_DECJ).dms[2] * 1000.0
    VLBI_DECJ_err_ms = VLBI_pos_SSB_err.dec.dms[2] * 1000.0

    x, y = pdf_values(x0=VLBI_deltaDECJ_ms, uL=VLBI_DECJ_err_ms, uR=VLBI_DECJ_err_ms)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=2)

    # timing
    if timing_source == "NG":
        timing_deltaDECJ_ms = (Angle(eq_timing_model.DECJ.quantity) - ref_DECJ).dms[2] * 1000.0
        timing_DECJ_err_ms = Angle(eq_timing_model.DECJ.uncertainty).dms[2] * 1000.0
    elif timing_source == "PPTA":
        timing_deltaDECJ_ms = (timing_pos_SSB.dec.to(u.degree) - ref_DECJ).dms[2] * 1000.0
        timing_DECJ_err_ms = timing_pos_SSB_err.dec.dms[2] * 1000.0

    x, y = pdf_values(x0=timing_deltaDECJ_ms, uL=timing_DECJ_err_ms, uR=timing_DECJ_err_ms)
#    x = [np.float64(z) for z in x]
#    y = [np.float64(z) for z in y]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i + 1, col=2)
    fig.update_xaxes(title_text="$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$", row=i+1, col=2)

    #------------------------------Parallax------------------------------
    # VLBI
    x, y = pdf_values(x0=VLBI_data.loc[PSR, "VLBI_PX"], uL=VLBI_data.loc[PSR, "VLBI_PX_uL"], uR=VLBI_data.loc[PSR, "VLBI_PX_uR"])
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=3)

    # Timing
    if timing_source == "NG":
        timing_PX = eq_timing_model.PX.quantity
        timing_PX_err = eq_timing_model.PX.uncertainty
    elif timing_source == "PPTA":
        timing_PX = timing_data.loc[PSR, "timing_PX"]
        timing_PX_err = timing_data.loc[PSR, "timing_PX_uL"],

#    if PSR == "J0437-4715":
#        timing_PX = 6.65
#        timing_PX_err = 0.51

    x, y = pdf_values(x0=timing_PX, uL=timing_PX_err, uR=timing_PX_err)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=3)
    fig.update_xaxes(title_text="$\Pi [\mathrm{mas}]$", row=i+1, col=3)


    #------------------------------Proper Motion------------------------------
    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(VLBI_data.loc[PSR, "VLBI_PMRA"], VLBI_data.loc[PSR, "VLBI_PMRA_" + error_side])
        VLBI_PMDEC = ufloat(VLBI_data.loc[PSR, "VLBI_PMDEC"], VLBI_data.loc[PSR, "VLBI_PMDEC_" + error_side])

        # As far as I can tell, all the papers where I extracted the values of PMRA already include the cos(delta) in
        # the definition for PMRA. That is, PMRA = dalpha/dt * cos(delta)
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2)

        if error_side=="uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side=="uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    x, y = pdf_values(x0=VLBI_PM.nominal_value, uL=VLBI_PM_uL, uR=VLBI_PM_uR)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=4)
#    fig.add_vline(x=VLBI_PM.nominal_value - 3 * VLBI_PM.std_dev, line_width=2, line_dash="dash", line_color=VLBI_color,
#                  timing_solution=i + 1, col=4)
#    fig.add_vline(x=VLBI_PM.nominal_value + 3 * VLBI_PM.std_dev, line_width=2, line_dash="dash", line_color=VLBI_color,
#                  timing_solution=i + 1, col=4)

    # Timing
    if timing_source == "NG":
        timing_PMRA = ufloat(eq_timing_model.PMRA.value, eq_timing_model.PMRA.uncertainty.value)
        timing_PMDEC = ufloat(eq_timing_model.PMDEC.value, eq_timing_model.PMDEC.uncertainty.value)
    elif timing_source == "PPTA":
        timing_PMRA = ufloat(timing_data.loc[PSR, "timing_PMRA"], timing_data.loc[PSR, "timing_PMRA_uL"])
        timing_PMDEC = ufloat(timing_data.loc[PSR, "timing_PMDEC"], timing_data.loc[PSR, "timing_PMDEC_uL"])

    # As far as I can tell, all the papers where I extracted the values of PMRA already include the cos(delta) in
    # the definition for PMRA. That is, PMRA = dalpha/dt * cos(delta)
    timing_PM = umath.sqrt(timing_PMDEC ** 2 + timing_PMRA ** 2)

    x, y = pdf_values(x0=timing_PM.nominal_value, uL=timing_PM.std_dev, uR=timing_PM.std_dev)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=4)
    fig.update_xaxes(title_text="$\mu~[\mathrm{mas~yr^{-1}}]$", row=i+1, col=4)

    # ------------------------------PMRA------------------------------
    # VLBI
    x, y = pdf_values(x0=VLBI_data.loc[PSR, "VLBI_PMRA"], uL=VLBI_data.loc[PSR, "VLBI_PMRA_uL"],
                      uR=VLBI_data.loc[PSR, "VLBI_PMRA_uR"])
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i + 1,
                  col=5)

    # Timing
    x, y = pdf_values(x0=timing_PMRA.nominal_value, uL=timing_PMRA.std_dev, uR=timing_PMRA.std_dev)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i + 1, col=5)
    fig.update_xaxes(title_text="$\mu_{\mathrm{RA}}~[\mathrm{mas~yr^{-1}}]$", row=i + 1, col=5)

    # ------------------------------PMDEC------------------------------
    # VLBI
    x, y = pdf_values(x0=VLBI_data.loc[PSR, "VLBI_PMDEC"], uL=VLBI_data.loc[PSR, "VLBI_PMDEC_uL"],
                      uR=VLBI_data.loc[PSR, "VLBI_PMDEC_uR"])
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i + 1,
                  col=6)

    # Timing
    x, y = pdf_values(x0=timing_PMDEC.nominal_value, uL=timing_PMDEC.std_dev, uR=timing_PMDEC.std_dev)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False),
                  row=i + 1, col=6)


    fig.update_xaxes(title_text="$\mu_{\mathrm{DEC}}~[\mathrm{mas~yr^{-1}}]$", row=i + 1, col=6)



    fig.update_yaxes(title_text=PSR, row=i+1, col=1)

fig.update_layout(
    title_text="Timing vs VLBI Astrometric Parameters",
    title_font=dict(size=20),
    title_x=0.5,
    title_y=0.98,
    title_xanchor="center",
    title_yanchor="top"
)

fig.update_xaxes(automargin=True)
fig.update_yaxes(automargin=True)
fig.show()
#fig.write_html("./figures/astrometric_comparison_frame_tie.html")
fig.write_image("./figures/astrometric_comparison_frame_tie_Wang.png", width=1200, height=4800)
