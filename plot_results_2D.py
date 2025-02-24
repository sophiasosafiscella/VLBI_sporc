import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pint.models import get_model
from scipy.interpolate import griddata
import glob

def plot_contour(df, tm, timing_data, x_col, y_col, w_col, ax):

    params = ["RAJ", "DECJ", "PX", "PMRA", "PMDEC"]
    # Extract columns
    x = df[x_col]
    y = df[y_col]
    w = df[w_col]

    # Create grid values first
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate w values on grid
    zi = griddata((x, y), w, (xi, yi), method='linear')

    # Plot contour
    contour = ax.contourf(xi, yi, zi, levels=15, cmap="viridis")
    plt.colorbar(contour, ax=ax, label=w_col)

    # Extract the reference timing values
    ax.scatter(timing_data[x_col], timing_data[y_col], marker='x', c='red', s=400)

    ax.set_xlabel(f"{x_col} [{getattr(tm, x_col).units}]")
    ax.set_ylabel(f"{y_col} [{getattr(tm, y_col).units}]")
#    ax.set_title(f'{x_col} vs {y_col} with {w_col} as color')


if __name__ == "__main__":

    PSR_name: str = "J0030+0451"
    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

    # Load the timing solution
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0)

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)
    result_df = result_df.rename(columns={'PMRA': 'PX', 'PMDEC': 'PMRA', 'PX': 'PMDEC'})

    # Find the solution with the highest posterior
    best_sol_idx = result_df['posterior'].idxmax()
    best_sol = result_df.iloc[best_sol_idx].to_dict()

    # Create subplots
    sns.set_context('poster')
    sns.set_style('ticks')
    fig, axs = plt.subplots(5, 5, figsize=(18, 12))
    fig.suptitle(PSR_name)

    for row in range(5):
        for col in range(row+1, 5):
            axs[row, col].axis('off')

    # Plot each pair
    plot_contour(df, eq_timing_model, timing_data, 'RAJ', 'DECJ', 'posterior', axs[0, 0])
    plot_contour(df, eq_timing_model, timing_data, 'RAJ', 'PMRA', 'posterior', axs[1, 0])
    plot_contour(df, eq_timing_model, timing_data, 'RAJ', 'PMDEC', 'posterior', axs[2, 0])
    plot_contour(df, eq_timing_model, timing_data, 'RAJ', 'PX', 'posterior', axs[3, 0])
    plot_contour(df, eq_timing_model, timing_data, 'RAJ', 'PX', 'posterior', axs[4, 0])
    plot_contour(df, eq_timing_model, timing_data,'PMDEC', 'PX', 'posterior', axs[1, 1])

    plt.tight_layout()
    plt.show()
