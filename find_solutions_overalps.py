import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from scipy.stats import norm
from pint import models

from astropy.coordinates import Angle, Longitude, Latitude, SkyCoord, FK5, ICRS, BarycentricMeanEcliptic, \
    BarycentricTrueEcliptic, Galactic, BaseEclipticFrame
from astropy.time import Time
import astropy.units as u
from VLBI_utils import pdf_value, plot_pdf

from shapely import Point, buffer, prepare, contains_properly
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon
from uncertainties import ufloat, umath, unumpy
from itertools import product
import math

import glob
import sys


def circle(radius, center):
    angle = np.linspace(0, 2 * np.pi, 150)
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]

    return np.column_stack((x, y))

def find_overlap(PSR_name, data, eq_timing_model, overlap_file):

    VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
    timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]
    grid_num: int = 10
    factor: int = 2.5

    # Timing model in ecliptical coordiantes
#    ec_timing_model = models.get_model(glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*.nb.par")[0])

    # Timing model in equatorial coordinates
#    eq_timing_model = ec_timing_model.as_ICRS(epoch=Time(data.loc[PSR_name, "POSEPOCH"], format="mjd"))

    #------------------------------Proper Motion------------------------------
    # Timing
    timing_PMRA = ufloat(eq_timing_model.PMRA.value, eq_timing_model.PMRA.uncertainty.value)
    timing_PMDEC = ufloat(eq_timing_model.PMDEC.value, eq_timing_model.PMDEC.uncertainty.value)

    timing_DECJ = ufloat(Angle(eq_timing_model.DECJ.quantity).rad, Angle(eq_timing_model.DECJ.uncertainty).rad)
    timing_PMRA_star = timing_PMRA * umath.cos(timing_DECJ)
    timing_PM = umath.sqrt(timing_PMRA_star ** 2 + timing_PMDEC ** 2)

    # Define grid points for X and Y
    x_values = np.linspace(timing_PMRA_star.nominal_value - 4 * timing_PMRA_star.std_dev,
                           timing_PMRA_star.nominal_value + 4 * timing_PMRA_star.std_dev, grid_num)
    y_values = np.linspace(timing_PMDEC.nominal_value - 4 * timing_PMDEC.std_dev,
                           timing_PMDEC.nominal_value + 4 * timing_PMDEC.std_dev, grid_num)

    # Calculate probability density functions (PDFs) for X and Y
    pdf_X = pdf_value(x_values, x0=timing_PMRA_star.nominal_value, uL=timing_PMRA_star.std_dev,
                      uR=timing_PMRA_star.std_dev)
    pdf_Y = pdf_value(y_values, x0=timing_PMDEC.nominal_value, uL=timing_PMDEC.std_dev, uR=timing_PMDEC.std_dev)

    # Calculate the joint probability distribution by multiplying the PDFs
    joint_pdf = np.outer(pdf_X, pdf_Y)

    # Normalize the joint PDF so that it sums to 1
    joint_pdf /= np.sum(joint_pdf)

    # Create a meshgrid for contour plotting
    X, Y = np.meshgrid(x_values, y_values)
    # Create figure and axes objects for main plot and marginal distributions
    sns.set_theme(context="paper", style="ticks", font_scale=1.8, rc={"axes.axisbelow": False, "grid.linewidth": 1.4})
    fig, ([ax_marginal_x, other], [ax_main, ax_marginal_y]) = plt.subplots(2, 2, figsize=(11, 8),
                                                                           gridspec_kw={'height_ratios': [1, 4],
                                                                                        'width_ratios': [4, 1],
                                                                                        'hspace': 0.00, 'wspace': 0.00})

    # Plot contour plot on main axes
    contour = ax_main.contourf(X, Y, joint_pdf, cmap="viridis", zorder=0)
    plt.colorbar(contour, ax=ax_main, label='Normalized Probability Density', location="left", pad=-0.15,
                 anchor=(-2.0, 0.5))
    ax_main.set_xlabel("$\mu_{\\alpha^{*}} = \mu_{\\alpha} \cos(\delta)$")
    ax_main.set_ylabel("$\mu_{\delta}$")

    # Create a polygon object containing the timing solution
    #    vertices = contour.collections[2].get_paths()[0].vertices
    timing_polygon = Polygon(contour.allsegs[1][0])
    prepare(timing_polygon)

    # Plot marginal distribution for X on top right subplot
    ax_marginal_x.plot(x_values, pdf_X, color='red')
    ax_marginal_x.set_xlim(ax_main.get_xlim())
    ax_marginal_x.set_ylim(0, np.max(pdf_X) * 1.1)
    ax_marginal_x.set_xticks([])
    #    ax_marginal_x.set_title('PDF of X')

    # Plot marginal distribution for Y on bottom left subplot
    ax_marginal_y.plot(pdf_Y, y_values, color='blue')
    ax_marginal_y.set_ylim(ax_main.get_ylim())
    ax_marginal_y.set_xlim(0, np.max(pdf_Y) * 1.1)
    ax_marginal_y.set_yticks([])
    #    ax_marginal_y.set_title('PDF of Y')

    # Remove unnecessary spines
    other.spines['right'].set_visible(False)
    other.spines['top'].set_visible(False)
    other.set_xticks([])
    other.set_yticks([])

    # Create a circle containing the +/- values
    circle_timing_out = plt.Circle((0, 0), radius=timing_PM.nominal_value + factor * timing_PM.std_dev, color='r', lw=3, ls=":",
                               fill=False,
                               label="$\mu_{\mathrm{timing}} \pm$" + str(factor) + "$\sigma_{\mu}$")
    circle_timing_in = plt.Circle((0, 0), radius=timing_PM.nominal_value - factor * timing_PM.std_dev, color='r', lw=3, ls=":",
                                fill=False)
    ax_main.add_patch(circle_timing_out)
    ax_main.add_patch(circle_timing_in)

    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    VLBI_DECJ = ufloat(Angle(data.loc[PSR_name, "VLBI_DECJ"]).rad, Angle(data.loc[PSR_name, "VLBI_DECJ_err"]).rad)

    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(data.loc[PSR_name, "VLBI_PMRA"], data.loc[PSR_name, "VLBI_PMRA_" + error_side])
        VLBI_PMDEC = ufloat(data.loc[PSR_name, "VLBI_PMDEC"], data.loc[PSR_name, "VLBI_PMDEC_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2 * (umath.cos(VLBI_DECJ) ** 2))

        if error_side == "VLBI_uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "VLBI_uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    # Make circles representing the PM +/- 2sigma region for VLBI
    circle_uL = plt.Circle((0, 0), VLBI_PM.nominal_value - factor * VLBI_PM.std_dev, color='g', lw=3, ls="--", fill=False,
                           label="$\mu_{\mathrm{VLBI}} \pm$" + str(factor) + "$\sigma_{\mu}$")
    circle_ML = plt.Circle((0, 0), VLBI_PM.nominal_value, color='b', lw=4, fill=False, label="$\mu_{\mathrm{VLBI}}$")
    circle_uR = plt.Circle((0, 0), VLBI_PM.nominal_value + factor * VLBI_PM.std_dev, color='g', lw=3, ls="--", fill=False)
    ax_main.add_patch(circle_uL)
    ax_main.add_patch(circle_ML)
    ax_main.add_patch(circle_uR)
    ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Adjust layout and show plot
    plt.suptitle(PSR_name)
    plt.tight_layout()
    plt.savefig("./figures/density_plots/" + PSR_name + "_PM.png", bbox_inches='tight')
    plt.show()

    # ------------------------------Parallax------------------------------

    fig = go.Figure()
    sns.set_theme(context="paper", style="dark", font_scale=1.5)
    # VLBI
    VLBI_x0: float = data.loc[PSR_name, "VLBI_PX"]
    VLBI_uL: float = data.loc[PSR_name, "VLBI_PX_uL"]
    VLBI_uR: float = data.loc[PSR_name, "VLBI_PX_uR"]

    x, y = plot_pdf(x0=VLBI_x0, uL=VLBI_uL, uR=VLBI_uR, num=grid_num)
    px_values = x
    fig.add_trace(go.Scatter(x=x, y=y, name="VLBI", fill='tozeroy', fillcolor=VLBI_color, mode='none'))
    fig.add_vline(x=VLBI_x0 - factor * VLBI_uL, line_width=3, line_dash="dash", line_color="green")
    fig.add_vline(x=VLBI_x0 + factor * VLBI_uR, line_width=3, line_dash="dash", line_color="green")

    # Timing
    timing_PX = eq_timing_model.PX.value
    timing_PX_err = eq_timing_model.PX.uncertainty.value

    if PSR_name == "J0437-4715":
        timing_PX = 6.65
        timing_PX_err = 0.51

    x, y = plot_pdf(x0=timing_PX, uL=timing_PX_err, uR=timing_PX_err, num=grid_num)
    fig.add_trace(go.Scatter(x=x, y=y, name="Timing", fill='tozeroy', fillcolor=timing_color, mode='none'))
    fig.add_vline(x=timing_PX-2*timing_PX_err, line_width=3, line_dash="dash", line_color="blue")
    fig.add_vline(x=timing_PX+2*timing_PX_err, line_width=3, line_dash="dash", line_color="blue")
    fig.update_xaxes(title_text="$\Pi [\mathrm{mas}]$")
    fig.write_image(f"./figures/density_plots/{PSR_name}_PX.png")
    fig.show()

    # Make an array to store the points that are in the overlap area
    results = np.empty((grid_num**3, 3))
    results[:, :] = np.nan

    # Iterate over all points
    for i, (mu_alpha_star, mu_delta, px) in enumerate(product(x_values, y_values, px_values)):

        if VLBI_PM.nominal_value - factor * VLBI_PM.std_dev < math.sqrt(mu_alpha_star**2 + mu_delta**2) < VLBI_PM.nominal_value + factor * VLBI_PM.std_dev:

#            if contains_properly(timing_polygon, Point(mu_alpha_star, mu_delta)):

             if np.max([timing_PX - factor * timing_PX_err, VLBI_x0 - factor * VLBI_uL]) < px < np.min([timing_PX + factor * timing_PX_err, VLBI_x0 + factor * VLBI_uR]):

                 results[i, :] = [mu_alpha_star / math.cos(timing_DECJ.nominal_value), mu_delta, px]

    overlap_df = pd.DataFrame(data=results, columns=["PMRA", "PMDEC", "PX"]).dropna(how="any", ignore_index=True)
    overlap_df.to_csv(overlap_file, sep=" ", header=True, index_label="ArrayTaskID")

    return


