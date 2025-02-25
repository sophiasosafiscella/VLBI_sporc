import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from astropy.coordinates import Angle
import astropy.units as u
import sys

PSR_names = ["J0030+0451", "J2145-0750", "J2010-1323"]
VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

# Load the timing solution
timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0)

fig = make_subplots(rows=3, cols=5)

for row, PSR_name in enumerate(PSR_names):

    print(PSR_name)

    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"

    # Load the posteriors
    df = pd.read_pickle(posteriors_file)

    # Convert PX, PMRA, PMDEC to float
    df[["PX", "PMRA", "PMDEC", "posterior"]] = df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # Find the solution with the highest posterior
    best_sol_idx = df['posterior'].idxmax()
    best_sol = df.loc[best_sol_idx].to_dict()

    # RAJ
    sols = df[(df['PX'] == best_sol['PX']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PMDEC'] == best_sol['PMDEC'])]

    timing_RAJ = Angle(timing_astrometric_data.loc[PSR_name, "ra_t"], unit=u.hourangle)
    ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")

    timing_deltaRAJ_ms = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
    deltaRAJ_ms = [(Angle(x, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0 for x in sols["RAJ"]]

    fig.add_trace(go.Scatter(x=deltaRAJ_ms, y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)),
                  row=row+1, col=1)
    fig.add_vline(x=timing_deltaRAJ_ms, line_width=3, line_dash="dash", line_color=timing_color, row=row+1, col=1)
    fig.update_yaxes(showticklabels=False, row=row+1, col=1)
    fig.update_xaxes(title_text="$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$", row=row+1, col=1)

    # DECJ
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['PX'] == best_sol['PX']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PMDEC'] == best_sol['PMDEC'])]

    timing_DECJ = Angle(timing_astrometric_data.loc[PSR_name, "dec_t"], unit=u.degree)
    ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 1)}s")

    timing_deltaDECJ_ms = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
    deltaDECJ_ms = [(Angle(x, unit=u.degree) - ref_DECJ).dms[2] * 1000.0 for x in sols["DECJ"]]

    fig.add_trace(go.Scatter(x=deltaDECJ_ms, y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)),
                  row=row+1, col=2)
    fig.add_vline(x=timing_deltaDECJ_ms, line_width=3, line_dash="dash", line_color=timing_color, row=row+1, col=2)
    fig.update_yaxes(showticklabels=False, row=row+1, col=2)
    fig.update_xaxes(title_text="$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$", row=row+1, col=2)

    # PX
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PMDEC'] == best_sol['PMDEC'])]

    fig.add_trace(go.Scatter(x=sols["PX"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=3)
    fig.add_vline(x=timing_astrometric_data.loc[PSR_name, "px_t"], line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=3)
    fig.update_yaxes(showticklabels=False, row=row+1, col=3)
    fig.update_xaxes(title_text="PX [mas]", row=row+1, col=3)

    # PMRA
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PX'] == best_sol['PX']) & (df['PMDEC'] == best_sol['PMDEC'])]

    fig.add_trace(go.Scatter(x=sols["PMRA"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=4)
    fig.add_vline(x=timing_astrometric_data.loc[PSR_name, "pmra_t"], line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=4)
    fig.update_yaxes(showticklabels=False, row=row+1, col=4)
    fig.update_xaxes(title_text="PMRA [mas/yr]", row=row+1, col=4)

    # PMDEC
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PX'] == best_sol['PX'])]

    fig.add_trace(go.Scatter(x=sols["PMDEC"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=5)
    fig.add_vline(x=timing_astrometric_data.loc[PSR_name, "pmdec_t"], line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=5)
    fig.update_yaxes(showticklabels=False, row=row+1, col=5)
    fig.update_xaxes(title_text="PMDEC [mas/yr]", row=row+1, col=5)

    fig.update_yaxes(title_text=PSR_name, row=row + 1, col=1)

    fig.update_xaxes(tickformat=".3f")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        autosize=False,
        width=1200,
        height=700,
    )

fig.show()
