import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import astropy.units as u
from astropy.coordinates import Angle

from pint.models import get_model

from VLBI_utils import pdf_values

from shapely import Point, buffer, prepare, contains_properly
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon
from uncertainties import ufloat, umath, unumpy
from itertools import product
from math import sqrt

import glob


def circle(radius, center):
    angle = np.linspace(0, 2 * np.pi, 150)
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]

    return np.column_stack((x, y))


def overlap_range(mu1, sigma1, mu2, sigma2, factor, grid_num):
    # Calculate the bounds for both distributions
    lower1, upper1 = mu1 - factor * sigma1, mu1 + factor * sigma1
    lower2, upper2 = mu2 - factor * sigma2, mu2 + factor * sigma2

    # Calculate the overlap range
    lower_overlap = max(lower1, lower2)
    upper_overlap = min(upper1, upper2)

    # Check if there is an overlap
    if lower_overlap < upper_overlap:
        return (lower_overlap, upper_overlap), np.linspace(lower_overlap, upper_overlap, grid_num)
    else:
        return None, None # No overlap

def asymmetrical_overlap_range(mu1, sigma1, mu2, sigma2L, sigma2R, factor, grid_num):
    # Calculate the bounds for both distributions
    lower1, upper1 = mu1 - factor * sigma1, mu1 + factor * sigma1
    lower2, upper2 = mu2 - factor * sigma2L, mu2 + factor * sigma2R

    # Calculate the overlap range
    lower_overlap = max(lower1, lower2)
    upper_overlap = min(upper1, upper2)

    # Check if there is an overlap
    if lower_overlap < upper_overlap:
        return (lower_overlap, upper_overlap), np.linspace(lower_overlap, upper_overlap, grid_num)
    else:
        return None, None # No overlap


def pm_overlap_range(mu_PMRA_timing, u_PMRA_timing, mu_PMDEC_timing, u_PMDEC_timing, mu_PM_VLBI, uL_PM_VLBI, uR_PM_VLBI, n=3, grid_num=100):
    # Calculate the bounds for both distributions
    lower_PMRA_timing, upper_PMRA_timing = mu_PMRA_timing - n * u_PMRA_timing, mu_PMRA_timing + n * u_PMRA_timing
    lower_PMDEC_timing, upper_PMDEC_timing = mu_PMDEC_timing - n * u_PMDEC_timing, mu_PMDEC_timing + n * u_PMDEC_timing
    lower_PM_VLBI, upper_PM_VLBI = mu_PM_VLBI - n * uL_PM_VLBI, mu_PM_VLBI + n * uR_PM_VLBI

    pmra_list = np.linspace(lower_PMRA_timing, upper_PMRA_timing, grid_num)
    pmdec_list = np.linspace(lower_PMDEC_timing, upper_PMDEC_timing, grid_num)

    overlaps = [(var1, var2) for var1, var2 in product(pmra_list, pmdec_list) if lower_PM_VLBI ** 2 <= var1 ** 2 + var2 ** 2 <= upper_PM_VLBI ** 2]

    if len(overlaps) > 0:
        return True, overlaps
    else:
        return None, None


def find_solutions(PSR_name, data, eq_timing_model, factor: int = 3, grid_num: int = 10, plot=False):
    VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
    timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

    # ------------------------------RAJ------------------------------
    timing_RAJ = ufloat(Angle(eq_timing_model.RAJ.quantity).rad, Angle(eq_timing_model.RAJ.uncertainty).rad)
    VLBI_RAJ = ufloat(Angle(data.loc[PSR_name, "VLBI_RAJ"]).rad, Angle(data.loc[PSR_name, "VLBI_RAJ_err"]).rad)

    RAJ_overlap, RAJ_values = overlap_range(timing_RAJ.nominal_value, timing_RAJ.std_dev,
                                            VLBI_RAJ.nominal_value, VLBI_RAJ.std_dev,  factor, grid_num)

    # ------------------------------DECJ------------------------------
    timing_DECJ = ufloat(Angle(eq_timing_model.DECJ.quantity).rad, Angle(eq_timing_model.DECJ.uncertainty).rad)
    VLBI_DECJ = ufloat(Angle(data.loc[PSR_name, "VLBI_DECJ"]).rad, Angle(data.loc[PSR_name, "VLBI_DECJ_err"]).rad)

    DECJ_overlap, DECJ_values = overlap_range(timing_DECJ.nominal_value, timing_DECJ.std_dev,
                                              VLBI_DECJ.nominal_value, VLBI_DECJ.std_dev, factor, grid_num)

    '''
    # ------------------------------PMRA------------------------------
    PMRA_overlap, PMRA_values = asymmetrical_overlap_range(eq_timing_model.PMRA.value,
                                                           eq_timing_model.PMRA.uncertainty.value,
                                                           eq_timing_model.PMRA.uncertainty.value,
                                                           data.loc[PSR_name, "VLBI_PMRA"],
                                                           data.loc[PSR_name, "VLBI_PMRA_uL"],
                                                           data.loc[PSR_name, "VLBI_PMRA_uR"], factor, grid_num)

    # ------------------------------PMDEC------------------------------
    PMDEC_overlap, PMDEC_values = asymmetrical_overlap_range(eq_timing_model.PMDEC.value,
                                                           eq_timing_model.PMDEC.uncertainty.value,
                                                           eq_timing_model.PMDEC.uncertainty.value,
                                                           data.loc[PSR_name, "VLBI_PMDEC"],
                                                           data.loc[PSR_name, "VLBI_PMDEC_uL"],
                                                           data.loc[PSR_name, "VLBI_PMDEC_uR"], factor, grid_num)
    '''

    # ------------------------------Proper Motion------------------------------
    # VLBI: sometimes the error bars are asymmetric. We will propagate errors twice, each time assuming a symmetric
    # error equal to either VLBI_uL or VLBI_uR:

    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(data.loc[PSR_name, "VLBI_PMRA"], data.loc[PSR_name, "VLBI_PMRA_" + error_side])
        VLBI_PMDEC = ufloat(data.loc[PSR_name, "VLBI_PMDEC"], data.loc[PSR_name, "VLBI_PMDEC_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2)

        if error_side == "uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    # Find the overlaps

    PM_overlap, PM_values = pm_overlap_range(eq_timing_model.PMRA.value, eq_timing_model.PMRA.uncertainty.value,
                                   eq_timing_model.PMDEC.value, eq_timing_model.PMDEC.uncertainty.value,
                                   VLBI_PM, VLBI_PM_uL, VLBI_PM_uR)

    if RAJ_overlap and DECJ_overlap and plot:
        timing_PMRA = ufloat(eq_timing_model.PMRA.value, eq_timing_model.PMRA.uncertainty.value)
        timing_PMDEC = ufloat(eq_timing_model.PMDEC.value, eq_timing_model.PMDEC.uncertainty.value)
        timing_PM = umath.sqrt(timing_PMRA ** 2 + timing_PMDEC ** 2)

        # Make a grid of PMRA/PMDEC values, and calculate probability density functions (PDFs) for each of those values
        PMRA_values, PMRA_pdf = pdf_values(x0=timing_PMRA.nominal_value, uL=timing_PMRA.std_dev,
                                           uR=timing_PMRA.std_dev, factor=factor, num=grid_num)
        PMDEC_values, PMDEC_pdf = pdf_values(x0=timing_PMDEC.nominal_value, uL=timing_PMDEC.std_dev,
                                             uR=timing_PMDEC.std_dev, factor=factor, num=grid_num)

        # Calculate the joint probability distribution by multiplying the PDFs
        joint_pdf = np.outer(PMRA_pdf, PMDEC_pdf)

        # Normalize the joint PDF so that it sums to 1
        joint_pdf /= np.sum(joint_pdf)

        # Create a meshgrid for contour plotting
        X, Y = np.meshgrid(PMRA_values, PMDEC_values)

        # Create figure and axes objects for main plot and marginal distributions
        sns.set_theme(context="paper", style="ticks", font_scale=1.8,
                      rc={"axes.axisbelow": False, "grid.linewidth": 1.4})
        fig, ([ax_marginal_x, other], [ax_main, ax_marginal_y]) = plt.subplots(2, 2, figsize=(11, 8),
                                                                               gridspec_kw={'height_ratios': [1, 4],
                                                                                            'width_ratios': [4, 1],
                                                                                            'hspace': 0.00,
                                                                                            'wspace': 0.00})

        # Plot contour plot on main axes
        contour = ax_main.contourf(X, Y, joint_pdf, cmap="viridis", zorder=0)
        plt.colorbar(contour, ax=ax_main, label='Normalized Probability Density', location="left", pad=-0.15,
                     anchor=(-2.0, 0.5))
        ax_main.set_xlabel("$\mu_{\\alpha^{*}} = \mu_{\\alpha} \cos(\delta)$")
        ax_main.set_ylabel("$\mu_{\delta}$")

        # Create a polygon object containing the timing solution
        timing_polygon = Polygon(contour.allsegs[1][0])
        prepare(timing_polygon)

        # Plot marginal distribution for X on top right subplot
        ax_marginal_x.plot(PMRA_values, PMRA_pdf, color='red')
        ax_marginal_x.set_xlim(ax_main.get_xlim())
        ax_marginal_x.set_ylim(0, np.max(PMRA_pdf) * 1.1)
        ax_marginal_x.set_xticks([])
        #    ax_marginal_x.set_title('PDF of X')

        # Plot marginal distribution for Y on bottom left subplot
        ax_marginal_y.plot(PMDEC_pdf, PMDEC_values, color='blue')
        ax_marginal_y.set_ylim(ax_main.get_ylim())
        ax_marginal_y.set_xlim(0, np.max(PMDEC_pdf) * 1.1)
        ax_marginal_y.set_yticks([])
        #    ax_marginal_y.set_title('PDF of Y')

        # Remove unnecessary spines
        other.spines['right'].set_visible(False)
        other.spines['top'].set_visible(False)
        other.set_xticks([])
        other.set_yticks([])

        # Create a circle containing the +/- values
        circle_timing_out = plt.Circle((0, 0), radius=timing_PM.nominal_value + factor * timing_PM.std_dev, color='r',
                                       lw=3, ls=":",
                                       fill=False,
                                       label="$\mu_{\mathrm{timing}} \pm$" + str(factor) + "$\sigma_{\mu}$")
        circle_timing_in = plt.Circle((0, 0), radius=timing_PM.nominal_value - factor * timing_PM.std_dev, color='r',
                                      lw=3, ls=":",
                                      fill=False)
        ax_main.add_patch(circle_timing_out)
        ax_main.add_patch(circle_timing_in)

        # Make circles representing the PM +/- sigma region for VLBI
        circle_uL = plt.Circle((0, 0), VLBI_PM.nominal_value - factor * VLBI_PM.std_dev, color='g', lw=3, ls="--",
                               fill=False,
                               label="$\mu_{\mathrm{VLBI}} \pm$" + str(factor) + "$\sigma_{\mu}$")
        circle_ML = plt.Circle((0, 0), VLBI_PM.nominal_value, color='b', lw=4, fill=False,
                               label="$\mu_{\mathrm{VLBI}}$")
        circle_uR = plt.Circle((0, 0), VLBI_PM.nominal_value + factor * VLBI_PM.std_dev, color='g', lw=3, ls="--",
                               fill=False)
        ax_main.add_patch(circle_uL)
        ax_main.add_patch(circle_ML)
        ax_main.add_patch(circle_uR)
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Adjust layout and show plot
        plt.suptitle(PSR_name)
        plt.tight_layout()
        plt.savefig("./results/frame_tie/" + PSR_name + "_PM.png", bbox_inches='tight')
        plt.show()

    # ------------------------------Parallax------------------------------
    VLBI_DECJ = ufloat(Angle(data.loc[PSR_name, "VLBI_DECJ"]).rad, Angle(data.loc[PSR_name, "VLBI_DECJ_err"]).rad)

    PX_overlap, PX_values = asymmetrical_overlap_range(eq_timing_model.PX.value, eq_timing_model.PX.uncertainty.value,
                                                       data.loc[PSR_name, "VLBI_PX"], data.loc[PSR_name, "VLBI_PX_uL"],
                                                       data.loc[PSR_name, "VLBI_PX_uR"], factor, grid_num)

    if RAJ_overlap and DECJ_overlap and plot:
        fig = go.Figure()
        sns.set_theme(context="paper", style="dark", font_scale=1.5)

        timing_PX = eq_timing_model.PX.value
        timing_PX_err = eq_timing_model.PX.uncertainty.value
        PX_values, PX_pdf = pdf_values(x0=timing_PX, uL=timing_PX_err, uR=timing_PX_err, factor=factor, num=grid_num)

        VLBI_PX_x0: float = data.loc[PSR_name, "VLBI_PX"]
        VLBI_PX_uL: float = data.loc[PSR_name, "VLBI_PX_uL"]
        VLBI_PX_uR: float = data.loc[PSR_name, "VLBI_PX_uR"]

        fig.add_trace(
            go.Scatter(x=PX_values, y=PX_pdf, name="Timing", fill='tozeroy', fillcolor=timing_color, mode='none'))
        fig.add_vline(x=timing_PX - 2 * timing_PX_err, line_width=3, line_dash="dash", line_color="blue")
        fig.add_vline(x=timing_PX + 2 * timing_PX_err, line_width=3, line_dash="dash", line_color="blue")

        x, y = pdf_values(x0=VLBI_PX_x0, uL=VLBI_PX_uL, uR=VLBI_PX_uR, factor=factor, num=grid_num)
        fig.add_trace(go.Scatter(x=x, y=y, name="VLBI", fill='tozeroy', fillcolor=VLBI_color, mode='none'))
        fig.add_vline(x=VLBI_PX_x0 - factor * VLBI_PX_uL, line_width=3, line_dash="dash", line_color="green")
        fig.add_vline(x=VLBI_PX_x0 + factor * VLBI_PX_uR, line_width=3, line_dash="dash", line_color="green")

        fig.update_xaxes(title_text="$\Pi [\mathrm{mas}]$")
        fig.write_image(f"./results/frame_tie/{PSR_name}_PX.png")
        fig.show()

    # ------------------------------Find the overlap------------------------------

    if RAJ_overlap and DECJ_overlap and PM_overlap and PX_overlap:

        return product(RAJ_values, DECJ_values, PM_values, PX_values)

    else:
        return None


if __name__ == "__main__":

    # File containing the timing and VLBI astrometric VLBI_data
    VLBI_astrometric_data = pd.read_csv("./data/VLBI_astrometric_values.csv", index_col=0)
    PSR_list = VLBI_astrometric_data.index  # List of pulsars

    for PSR_name in PSR_list:

        print(f"Finding the possible timing solutions for {PSR_name}")

        # Names of the .tim and .par files
        timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
        parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

        # Load the timing model and convert to equatorial coordinates
        ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
        eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

        # FIND THE OVERLAP BETWEEN THE TIMING AND VLBI SOLUTIONS
        solutions = find_solutions(PSR_name, VLBI_astrometric_data, eq_timing_model, grid_num=20, plot=True)

        if solutions:
            overlap_df = pd.DataFrame(data=solutions, columns=["RA", "DEC", "PM", "PX"])
            overlap_df.to_pickle(f"./results/frame_tie/{PSR_name}_overlap_frame_tie.pkl")
