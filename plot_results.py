import pandas as pd
import plotly.express as px
import sys

PSR_name: str = "J0030+0451"
posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

pars = ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"]

# Load the posteriors
result_df = pd.read_pickle(posteriors_file)

# Find the solution with the highest posterior
best_sol_idx = result_df['posterior'].idxmax()
best_sol = result_df.iloc[best_sol_idx].to_dict()

sols = result_df[(result_df['RAJ'] == best_sol['RAJ']) & (result_df['DECJ'] == best_sol['DECJ']) &
                 (result_df['PMRA'] == best_sol['PMRA']) & (result_df['PMDEC'] == best_sol['PMDEC'])]

print(sols)

