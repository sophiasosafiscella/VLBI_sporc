import numpy as np
import pandas as pd
from glob import glob
import sys
import os

PSR_name: str = sys.argv[1]
posteriors_dir: str = f"./results/timing_posteriors/{PSR_name}"
posterior_files = glob.glob(f"{posteriors_dir}/*posterior.pkl")

# Array set to zero but to be replaced with the posteriors
res_np = np.zeros((len(posterior_files), 4), dtype=float)

for i, file in enumerate(posterior_files):
    res_np[i, :] = np.load(file)
    os.remove(file)

# Save the results to a DataFrame
result_df = pd.DataFrame(data=res_np, columns=["PMRA", "PMDEC", "PX", "posterior"])
result_df.to_pickle(f"./results/{PSR_name}/{PSR_name}timing_posteriors.pkl")
