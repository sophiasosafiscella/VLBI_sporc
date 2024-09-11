import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pint.models import get_model
from scipy.interpolate import griddata
import glob

def plot_contour(df, tm, x_col, y_col, w_col, ax):
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
    ax.scatter(getattr(tm, x_col).value, getattr(tm, y_col).value, marker='x', c='red', s=400)

    ax.set_xlabel(f"{x_col} [{getattr(tm, x_col).units}]")
    ax.set_ylabel(f"{y_col} [{getattr(tm, y_col).units}]")
#    ax.set_title(f'{x_col} vs {y_col} with {w_col} as color')


if __name__ == "__main__":

    PSR_name: str = "J1640+2224"

    # Get the nominal timing values
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Get the timing posteriors
    df = pd.read_pickle(f"./results/timing_posteriors/{PSR_name}_timing_posteriors.pkl").dropna(how='any')

    # Create subplots
    sns.set_context('poster')
    sns.set_style('ticks')
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(PSR_name)
    axs[0, 1].axis('off')

    # Plot each pair
    plot_contour(df, eq_timing_model, 'PMRA', 'PMDEC', 'posterior', axs[0, 0])
    plot_contour(df, eq_timing_model, 'PMRA', 'PX', 'posterior', axs[1, 0])
    plot_contour(df, eq_timing_model, 'PMDEC', 'PX', 'posterior', axs[1, 1])

    plt.tight_layout()
    plt.show()
