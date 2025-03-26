import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.coordinates import Angle
import astropy.units as u
from pint.models import get_model
from scipy.interpolate import griddata
from itertools import product
import glob
import sys

def find_timing_label(label):
    if label == 'RAJ':
        return 'ra_t'
    elif label == 'DECJ':
        return 'dec_t'
    elif label == 'PMRA':
        return 'pmra_t'
    elif label == 'PMDEC':
        return 'pmdec_t'
    elif label == 'PX':
        return 'px_t'

def find_timing_error_label(label):
    if label == 'RAJ':
        return 'ra_te'
    elif label == 'DECJ':
        return 'dec_te'
    elif label == 'PMRA':
        return 'pmra_te'
    elif label == 'PMDEC':
        return 'pmdec_te'
    elif label == 'PX':
        return 'px_te'

def plot_contour(df, best_sol, timing_astrometric_data, tm, x_label, y_label, ax):

    all_params = ["RAJ", "DECJ", "PX", "PMRA", "PMDEC"]
    p = [param for param in all_params if param not in (x_label, y_label)]

    sols = df[(df[p[0]] == best_sol[p[0]]) & (df[p[1]] == best_sol[p[1]]) & (df[p[2]] == best_sol[p[2]])]

    x_values = sols[x_label].unique()
    y_values = sols[y_label].unique()

    z_values = np.zeros((len(y_values), len(x_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            try:
                z_values[j, i] = sols[(sols[x_label] == x) & (sols[y_label] == y)]['posterior'].iloc[0]

            # This means that we couldn't find a timing solution for this combination
            except IndexError:
                z_values[j, i] = np.amin(sols['posterior'].to_numpy())

    # Create grid values first
#    xi = np.linspace(x.min(), x.max(), 100)
#    yi = np.linspace(y.min(), y.max(), 100)
#    xi, yi = np.meshgrid(xi, yi)

    # Interpolate w values on grid
#    zi = griddata((x, y), w, (xi, yi), method='linear')

    if x_label == 'RAJ':
        timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
        ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")
        x = (Angle(x_values, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        x_timing = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
        x_timing_error = Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0
        ax.set_xlabel("$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$")
    elif x_label == 'DECJ':
        timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
        ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 1)}s")
        x = (Angle(x_values, unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        x_timing = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
        x_timing_error = Angle(timing_astrometric_data['dec_te'], unit=u.degree).dms[2] * 1000.0
        ax.set_xlabel("$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$")
    else:
        x = x_values
        x_timing = timing_astrometric_data[find_timing_label(x_label)],
        x_timing_error = timing_astrometric_data[find_timing_error_label(x_label)]
        ax.set_xlabel(f"{x_label} [{getattr(tm, x_label).units}]")

    if y_label == 'RAJ':
        timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
        ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")
        y = (Angle(y_values, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        y_timing = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
        y_timing_error = Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0
        ax.set_ylabel("$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$")
    elif y_label == 'DECJ':
        timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
        ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 1)}s")
        y = (Angle(y_values, unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        y_timing = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
        y_timing_error = Angle(timing_astrometric_data['dec_te'], unit=u.degree).dms[2] * 1000.0
        ax.set_ylabel("$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$")
    else:
        y = y_values
        y_timing = timing_astrometric_data[find_timing_label(y_label)]
        y_timing_error = timing_astrometric_data[find_timing_error_label(y_label)]
        ax.set_ylabel(f"{y_label} [{getattr(tm, y_label).units}]")


    # Plot contour
    print(np.shape(z_values))
    contour = ax.contourf(x, y, z_values, levels=30, cmap="viridis")
    plt.colorbar(contour, ax=ax, label='posterior')

    # Extract the reference timing values
#    ax.scatter(x=x_timing, y=y_timing, marker='x', c='red', s=400)
#    ax.errorbar(x=x_timing, y=y_timing, xerr=x_timing_error, yerr=y_timing_error, marker='x', c='red')

#    ax.set_title(f'{x_col} vs {y_col} with {w_col} as color')


if __name__ == "__main__":

    PSR_name: str = "J0030+0451"
#    PSR_name: str = "J2145-0750"
    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

    # Get the nominal timing values
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    tm = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Load the timing solution
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0).loc[PSR_name]

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)
    print(f"Number of solutions = {len(result_df.index)}")

    # Convert PX, PMRA, PMDEC to float
    result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # Find the solution with the highest posterior
    best_sol_idx = result_df['posterior'].idxmax()
    best_sol = result_df.loc[best_sol_idx].to_dict()

    # Create subplots
    sns.set_context('poster')
    sns.set_style('ticks')
    fig, axs = plt.subplots(4, 4, figsize=(36, 30))
    fig.suptitle(PSR_name)

    for row in range(4):

        for col in range(row+1, 4):
            axs[row, col].axis('off')

        for col in range(4):
            if col > 0:
                axs[row, col].get_yaxis().set_visible(False)
            if row < 3:
                axs[row, col].get_xaxis().set_visible(False)



    # Plot each pair
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'DECJ', axs[0, 0])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PMRA', axs[1, 0])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PMRA', axs[1, 1])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PMDEC', axs[2, 0])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PMDEC', axs[2, 1])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMRA', 'PMDEC', axs[2, 2])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PX', axs[3, 0])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PX', axs[3, 1])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMRA', 'PX', axs[3, 2])
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMDEC', 'PX', axs[3, 3])

    plt.tight_layout()
    plt.show()
