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

def plot_contour(df, tm, x_label, y_label, ax):

#    all_params = ["RAJ", "DECJ", "PX", "PMRA", "PMDEC"]
#    p = [param for param in all_params if param not in (x_label, y_label)]

#    sols = df[(df[p[0]] == best_sol[p[0]]) & (df[p[1]] == best_sol[p[1]]) & (df[p[2]] == best_sol[p[2]])]

    x_values = df[x_label]
    y_values = df[y_label]
    z_values = df['posterior']

#    for i, x in enumerate(x_values):
#        for j, y in enumerate(y_values):
#            z_values[j, i] = df[(df[x_label] == x) & (df[y_label] == y)]['posterior'].sum()

#            if z_values[j, i] == 0.0:
#                z_values[j, i] = np.amin(df['posterior'].to_numpy())

    if x_label == 'RAJ':
        x_values = Angle(x_values, u.hourangle).rad
    elif x_label == 'DECJ':
        x_values = Angle(x_values, u.deg).rad

    if y_label == 'RAJ':
        y_values = Angle(y_values, u.hourangle).rad
    elif y_label == 'DECJ':
        y_values = Angle(y_values, u.deg).rad

    # Create grid values first
    xi = np.linspace(x_values.min(), x_values.max(), 100)
    yi = np.linspace(y_values.min(), y_values.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate w values on grid
    zi = griddata((x_values, y_values), z_values, (xi, yi), method='linear')

    # Plot contour
    contour = ax.contourf(xi, yi, zi, levels=15, cmap="viridis")
    plt.colorbar(contour, ax=ax, label='posterior')

    # Extract the reference timing values
#    ax.scatter(timing_data[x_col], timing_data[y_col], marker='x', c='red', s=400)

    ax.set_xlabel(f"{x_label} [{getattr(tm, x_label).units}]")
    ax.set_ylabel(f"{y_label} [{getattr(tm, y_label).units}]")
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
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0)

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)

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
    plot_contour(result_df, tm, 'RAJ', 'DECJ', axs[0, 0])
    plot_contour(result_df, tm, 'RAJ', 'PMRA', axs[1, 0])
    plot_contour(result_df, tm, 'DECJ', 'PMRA', axs[1, 1])
    plot_contour(result_df, tm, 'RAJ', 'PMDEC', axs[2, 0])
    plot_contour(result_df, tm, 'DECJ', 'PMDEC', axs[2, 1])
    plot_contour(result_df, tm, 'PMRA', 'PMDEC', axs[2, 2])
    plot_contour(result_df, tm, 'RAJ', 'PX', axs[3, 0])
    plot_contour(result_df, tm, 'DECJ', 'PX', axs[3, 1])
    plot_contour(result_df, tm, 'PMRA', 'PX', axs[3, 2])
    plot_contour(result_df, tm, 'PMDEC', 'PX', axs[3, 3])

    plt.tight_layout()
    plt.show()
