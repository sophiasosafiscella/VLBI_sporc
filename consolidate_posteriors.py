import numpy as np
import pandas as pd
from glob import glob
import sys
import os

PSR_name: str = sys.argv[1]
posteriors_dir: str = f"./results/timing_posteriors/{PSR_name}"
posterior_files = glob(f"{posteriors_dir}/*posterior.npy")

# Array set to zero but to be replaced with the posteriors
res_np = np.zeros((len(posterior_files), 3), dtype=float)
idx_np = np.zeros(len(posterior_files), dtype=int)

for i, file in enumerate(posterior_files):
    aux = np.load(file)
    res_np[i, :] = aux[1:]
    idx_np[i] = aux[0]
#    os.remove(file)

# Save the results to a DataFrame
result_df = pd.DataFrame(data=res_np, index=idx_np, columns=["PMRA", "PMDEC", "PX"]).sort_index()
result_df.to_pickle(f"./results/timing_posteriors/{PSR_name}_timing_posteriors.pkl")
