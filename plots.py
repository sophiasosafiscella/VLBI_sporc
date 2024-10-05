import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import norm
from pint import models

from astropy.coordinates import Angle, Longitude, Latitude, SkyCoord, FK5, ICRS, BarycentricMeanEcliptic, BarycentricTrueEcliptic, Galactic, BaseEclipticFrame
from astropy.time import Time
import astropy.units as u
#from skewnormal import pdf_values
from VLBI_utils import pdf_values

from uncertainties import ufloat, umath, unumpy

import glob
import sys

data = pd.read_csv("./data/astrometric_values.csv", index_col=0)
PSR_list = data.index
fig = make_subplots(rows=len(PSR_list), cols=4, vertical_spacing=0.02)
VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

for i, PSR in enumerate(PSR_list):

    ec_timing_model = models.get_model(glob.glob(f"./data/NG_15yr_dataset/par/{PSR}*.nb.par")[0])   # Ecliptical coordiantes
#    eq_timing_model = ec_timing_model.as_ICRS()  # Equatorial coordinates
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # ------------------------------RAJ------------------------------
    VLBI_RAJ = Angle(data.loc[PSR, "VLBI_RAJ"])
    ref_RAJ = Angle(f"{int(VLBI_RAJ.hms[0])}h{int(VLBI_RAJ.hms[1])}m{int(VLBI_RAJ.hms[2])}s")

    # VLBI
    VLBI_deltaRAJ_ms = (Angle(data.loc[PSR, "VLBI_RAJ"]) - ref_RAJ).hms[2] * 1000.0
    VLBI_RAJ_err_ms = Angle(data.loc[PSR, "VLBI_RAJ_err"]).hms[2] * 1000.0
    x, y = pdf_values(x0=VLBI_deltaRAJ_ms, uL=VLBI_RAJ_err_ms, uR=VLBI_RAJ_err_ms)

    if i==0:
        fig.add_trace(go.Scatter(x=x, y=y, name="VLBI", fill='tozeroy', fillcolor=VLBI_color, mode='none'), row=i+1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=1)

    # timing
    timing_deltaRAJ_ms = (Angle(eq_timing_model.RAJ.quantity) - ref_RAJ).hms[2] * 1000.0
    timing_RAJ_err_ms = Angle(eq_timing_model.RAJ.uncertainty).hms[2] * 1000.0

    if PSR == "J0437-4715":
        timing_deltaRAJ_ms = (Angle("04h37m15.883185s") - ref_RAJ).hms[2] * 1000.0
        timing_RAJ_err_ms = Angle("0h0m0.000006s").hms[2] * 1000.0

    x, y = pdf_values(x0=timing_deltaRAJ_ms, uL=timing_RAJ_err_ms, uR=timing_RAJ_err_ms)
    x = [np.float64(z) for z in x]
    y = [np.float64(z) for z in y]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=1)

    if i==0:
        fig.add_trace(go.Scatter(x=x, y=y, name="Timing", fill='tozeroy', fillcolor=timing_color, mode='none'), row=i+1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=1)

    fig.update_xaxes(title_text="$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$", row=i+1, col=1)

    # ------------------------------DECJ------------------------------
    VLBI_DECJ = Angle(data.loc[PSR, "VLBI_DECJ"])
    ref_DECJ = Angle(f"{int(VLBI_DECJ.dms[0])}d{int(abs(VLBI_DECJ.dms[1]))}m{int(abs(VLBI_DECJ.dms[2]))}s")

    # VLBI
    VLBI_deltaDECJ_ms = (Angle(data.loc[PSR, "VLBI_DECJ"]) - ref_DECJ).dms[2] * 1000.0
    VLBI_DECJ_err_ms = Angle(data.loc[PSR, "VLBI_DECJ_err"]).dms[2] * 1000.0
    x, y = pdf_values(x0=VLBI_deltaDECJ_ms, uL=VLBI_DECJ_err_ms, uR=VLBI_DECJ_err_ms)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=2)

    # timing
    timing_deltaDECJ_ms = (Angle(eq_timing_model.DECJ.quantity) - ref_DECJ).dms[2] * 1000.0
    timing_DECJ_err_ms = Angle(eq_timing_model.DECJ.uncertainty).dms[2] * 1000.0

    x, y = pdf_values(x0=timing_deltaDECJ_ms, uL=timing_DECJ_err_ms, uR=timing_DECJ_err_ms)
    x = [np.float64(z) for z in x]
    y = [np.float64(z) for z in y]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i + 1, col=2)
    fig.update_xaxes(title_text="$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$", row=i+1, col=2)

    #------------------------------Parallax------------------------------
    # VLBI
    x, y = pdf_values(x0=data.loc[PSR, "VLBI_PX"], uL=data.loc[PSR, "VLBI_PX_uL"], uR=data.loc[PSR, "VLBI_PX_uR"])
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=VLBI_color, mode='none', showlegend=False), row=i+1, col=3)

    # Timing
    timing_PX = eq_timing_model.PX.quantity
    timing_PX_err = eq_timing_model.PX.uncertainty

    if PSR == "J0437-4715":
        timing_PX = 6.65
        timing_PX_err = 0.51

    x, y = pdf_values(x0=timing_PX, uL=timing_PX_err, uR=timing_PX_err)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=3)
    fig.update_xaxes(title_text="$\Pi [\mathrm{mas}]$", row=i+1, col=3)

    #------------------------------Proper Motion------------------------------
    VLBI_DECJ = ufloat(Angle(data.loc[PSR, "VLBI_DECJ"]).rad, Angle(data.loc[PSR, "VLBI_DECJ_err"]).rad)

    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(data.loc[PSR, "VLBI_PMRA"], data.loc[PSR, "VLBI_PMRA_" + error_side])
        VLBI_PMDEC = ufloat(data.loc[PSR, "VLBI_PMDEC"], data.loc[PSR, "VLBI_PMDEC_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2 * (umath.cos(VLBI_DECJ) ** 2))

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
    timing_PMRA = ufloat(eq_timing_model.PMRA.value, eq_timing_model.PMRA.uncertainty.value)
    timing_PMDEC = ufloat(eq_timing_model.PMDEC.value, eq_timing_model.PMDEC.uncertainty.value)
    timing_DECJ = ufloat(Angle(eq_timing_model.DECJ.quantity).rad, Angle(eq_timing_model.DECJ.uncertainty).rad)
    timing_PM = umath.sqrt(timing_PMDEC**2 + timing_PMRA**2 * (umath.cos(timing_DECJ)**2))
#    fig.add_vline(x=timing_PM.nominal_value-3*timing_PM.std_dev, line_width=1, line_dash="dash", line_color=timing_color, timing_solution=i+1, col=4)
#    fig.add_vline(x=timing_PM.nominal_value+3*timing_PM.std_dev, line_width=1, line_dash="dash", line_color=timing_color, timing_solution=i+1, col=4)

    x, y = pdf_values(x0=timing_PM.nominal_value, uL=timing_PM.std_dev, uR=timing_PM.std_dev)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=timing_color, mode='none', showlegend=False), row=i+1, col=4)
    fig.update_xaxes(title_text="$\mu~[\mathrm{mas~yr^{-1}}]$", row=i+1, col=4)

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
fig.write_html("./figures/astrometric_comparison.html")
fig.write_image("./figures/astrometric_comparison.png", width=1200, height=3600)
