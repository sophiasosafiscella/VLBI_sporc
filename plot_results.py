import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from astropy.coordinates import Angle
import astropy.units as u
from uncertainties import ufloat
import sys

PSR_names = ["J0030+0451", "J2145-0750", "J2010-1323"]
VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

# Load the timing solution
timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0)
VLBI_astrometric_data = pd.read_csv("./data/calibrated_vlbi_astrometric_data.csv", index_col=0, header=0)

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
    timing_RAJ_err = Angle(timing_astrometric_data.loc[PSR_name, "ra_te"], unit=u.hourangle).to_value(u.mas)

    VLBI_RAJ = Angle(VLBI_astrometric_data.loc[PSR_name, "ra_v"], unit=u.hourangle)
    VLBI_RAJ_err = Angle(VLBI_astrometric_data.loc[PSR_name, "ra_ve"], unit=u.hourangle).to_value(u.mas)

    ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")

    timing_deltaRAJ_ms = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
    VLBI_deltaRAJ_ms = (VLBI_RAJ - ref_RAJ).hms[2] * 1000.0
    deltaRAJ_ms = [(Angle(x, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0 for x in sols["RAJ"]]

    fig.add_trace(go.Scatter(x=deltaRAJ_ms, y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)),
                  row=row+1, col=1)

    fig.add_vline(x=timing_deltaRAJ_ms, line_width=3, line_dash="dash", line_color=timing_color, row=row + 1, col=1)
    fig.add_vrect(x0=timing_deltaRAJ_ms - 3 * timing_RAJ_err, x1=timing_deltaRAJ_ms + 3 * timing_RAJ_err, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row+1, col=1)

    fig.add_vline(x=VLBI_deltaRAJ_ms, line_width=3, line_dash="dash", line_color=VLBI_color, row=row + 1, col=1)
    fig.add_vrect(x0=VLBI_deltaRAJ_ms - 3 * VLBI_RAJ_err, x1=VLBI_deltaRAJ_ms + 3 * VLBI_RAJ_err, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row + 1, col=1)

    fig.update_yaxes(showticklabels=True, row=row+1, col=1)
    fig.update_xaxes(title_text="$\mathrm{RAJ} - " + f"{ref_RAJ:latex}"[1:-1] + " [\mathrm{mas}]$", row=row+1, col=1)

    # DECJ
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['PX'] == best_sol['PX']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PMDEC'] == best_sol['PMDEC'])]

    timing_DECJ = Angle(timing_astrometric_data.loc[PSR_name, "dec_t"], unit=u.degree)
    timing_DECJ_err = Angle(timing_astrometric_data.loc[PSR_name, "dec_te"], unit=u.degree).to_value(u.mas)

    VLBI_DECJ = Angle(VLBI_astrometric_data.loc[PSR_name, "dec_v"], unit=u.hourangle)
    VLBI_DECJ_err = Angle(VLBI_astrometric_data.loc[PSR_name, "dec_ve"], unit=u.hourangle).to_value(u.mas)

    ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 1)}s")

    timing_deltaDECJ_ms = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
    VLBI_deltaDECJ_ms = (VLBI_DECJ - ref_DECJ).dms[2] * 1000.0
    deltaDECJ_ms = [(Angle(x, unit=u.degree) - ref_DECJ).dms[2] * 1000.0 for x in sols["DECJ"]]

    fig.add_trace(go.Scatter(x=deltaDECJ_ms, y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)),
                  row=row+1, col=2)

    fig.add_vline(x=timing_deltaDECJ_ms, line_width=3, line_dash="dash", line_color=timing_color, row=row+1, col=2)
    fig.add_vrect(x0=timing_deltaDECJ_ms - 3 * timing_DECJ_err, x1=timing_deltaDECJ_ms + 3 * timing_DECJ_err, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row + 1, col=2)

    fig.add_vline(x=VLBI_deltaDECJ_ms, line_width=3, line_dash="dash", line_color=VLBI_color, row=row+1, col=2)
    fig.add_vrect(x0=VLBI_deltaDECJ_ms - 3 * VLBI_DECJ_err, x1=VLBI_deltaDECJ_ms + 3 * VLBI_DECJ_err, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row + 1, col=2)

    fig.update_yaxes(showticklabels=True, row=row+1, col=2)
    fig.update_xaxes(title_text="$\mathrm{DECJ} - (" + f"{ref_DECJ:latex}"[1:-1] + ") [\mathrm{mas}]$", row=row+1, col=2)

    # PX
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PMDEC'] == best_sol['PMDEC'])]

    timing_PX = ufloat(timing_astrometric_data.loc[PSR_name, "px_t"], timing_astrometric_data.loc[PSR_name, "px_te"])

    VLBI_PX = VLBI_astrometric_data.loc[PSR_name, "px_v"]
    VLBI_PX_uL = VLBI_astrometric_data.loc[PSR_name, "px_v_uL"]
    VLBI_PX_uR = VLBI_astrometric_data.loc[PSR_name, "px_v_uR"]

    fig.add_trace(go.Scatter(x=sols["PX"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=3)

    fig.add_vline(x=timing_PX.nominal_value, line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=3)
    fig.add_vrect(x0=timing_PX.nominal_value - 3 * timing_PX.std_dev, x1=timing_PX.nominal_value + 3 * timing_PX.std_dev, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row + 1, col=3)

    fig.add_vline(x=VLBI_PX, line_width=3, line_dash="dash", line_color=VLBI_color, row=row + 1, col=3)
    fig.add_vrect(x0=VLBI_PX - 3 * VLBI_PX_uL, x1=VLBI_PX + 3 * VLBI_PX_uR, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row + 1, col=3)

    fig.update_yaxes(showticklabels=True, row=row+1, col=3)
    fig.update_xaxes(title_text="PX [mas]", row=row+1, col=3)

    # PMRA
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PX'] == best_sol['PX']) & (df['PMDEC'] == best_sol['PMDEC'])]

    PMRA_timing = ufloat(timing_astrometric_data.loc[PSR_name, "pmra_t"], timing_astrometric_data.loc[PSR_name, "pmra_te"])

    VLBI_PMRA = VLBI_astrometric_data.loc[PSR_name, "pmra_v"]
    VLBI_PMRA_uL = VLBI_astrometric_data.loc[PSR_name, "pmra_v_uL"]
    VLBI_PMRA_uR = VLBI_astrometric_data.loc[PSR_name, "pmra_v_uR"]

    fig.add_trace(go.Scatter(x=sols["PMRA"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=4)

    fig.add_vline(x=timing_astrometric_data.loc[PSR_name, "pmra_t"], line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=4)
    fig.add_vrect(x0=PMRA_timing.nominal_value - 3 * PMRA_timing.std_dev, x1=PMRA_timing.nominal_value + 3 * PMRA_timing.std_dev, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row + 1, col=4)

    fig.add_vline(x=VLBI_PMRA, line_width=3, line_dash="dash", line_color=VLBI_color, row=row + 1, col=4)
    fig.add_vrect(x0=VLBI_PMRA - 3 * VLBI_PMRA_uL, x1=VLBI_PMRA + 3 * VLBI_PMRA_uR, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row + 1, col=4)

    fig.update_yaxes(showticklabels=True, row=row+1, col=4)
    fig.update_xaxes(title_text="PMRA [mas/yr]", row=row+1, col=4)

    # PMDEC
    sols = df[(df['RAJ'] == best_sol['RAJ']) & (df['DECJ'] == best_sol['DECJ']) &
              (df['PMRA'] == best_sol['PMRA']) & (df['PX'] == best_sol['PX'])]

    PMDEC_timing = ufloat(timing_astrometric_data.loc[PSR_name, "pmdec_t"], timing_astrometric_data.loc[PSR_name, "pmdec_te"])

    VLBI_PMDEC = VLBI_astrometric_data.loc[PSR_name, "pmdec_v"]
    VLBI_PMDEC_uL = VLBI_astrometric_data.loc[PSR_name, "pmdec_v_uL"]
    VLBI_PMDEC_uR = VLBI_astrometric_data.loc[PSR_name, "pmdec_v_uR"]

    fig.add_trace(go.Scatter(x=sols["PMDEC"], y=sols["posterior"], mode='lines+markers', marker=dict(color=VLBI_color)), row=row+1,
                  col=5)

    fig.add_vline(x=PMDEC_timing.nominal_value, line_width=3, line_dash="dash",
                  line_color=timing_color, row=row+1, col=5)
    fig.add_vrect(x0=PMDEC_timing.nominal_value - 3 * PMDEC_timing.std_dev, x1=PMDEC_timing.nominal_value + 3 * PMDEC_timing.std_dev, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row + 1, col=5)

    fig.add_vline(x=VLBI_PMDEC, line_width=3, line_dash="dash", line_color=VLBI_color, row=row+1, col=5)
    fig.add_vrect(x0=VLBI_PMDEC - 3 * VLBI_PMDEC_uL, x1=VLBI_PMDEC + 3 * VLBI_PMDEC_uR, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row + 1, col=5)

    fig.update_yaxes(showticklabels=True, row=row+1, col=5)
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
