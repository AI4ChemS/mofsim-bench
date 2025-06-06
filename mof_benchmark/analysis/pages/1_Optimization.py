import streamlit as st
import pandas as pd
import os


import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np

from mof_benchmark import base_dir

st.set_page_config(page_title="Optimization")
st.markdown("# Optimization analysis")

selected_calculator = None


# @st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_parquet(file_path)
    return df


# Load the data
optimization_results_file = (
    base_dir / "analysis" / "results" / "optimization_results.parquet"
)

optimization_rmsd = load_data(optimization_results_file)

if optimization_rmsd is None:
    st.markdown(
        f"File {optimization_results_file} not available . Make sure to run the analysis first."
    )

else:
    if "optimization_rmsd" not in st.session_state:
        st.session_state["optimization_rmsd"] = load_data(optimization_results_file)

    optimization_rmsd = st.session_state["optimization_rmsd"]
    optimization_rmsd["structure"] = optimization_rmsd.index.get_level_values(
        "structure"
    )
    optimization_rmsd["settings"] = optimization_rmsd.index.get_level_values("settings")
    optimization_rmsd["calculator"] = optimization_rmsd.index.get_level_values(
        "calculator"
    )

    selected_settings = st.sidebar.selectbox(
        "Settings",
        optimization_rmsd["settings"].unique(),
        format_func=lambda x: str(x).split("/")[-1],
    )

    selected_calculator = st.sidebar.selectbox(
        "Calculator",
        np.sort(
            optimization_rmsd[optimization_rmsd["settings"] == selected_settings][
                "calculator"
            ].unique()
        ),
    )

    @st.cache_data
    def filter_data(selected_settings, selected_calculator):

        if "optimization_rmsd" not in st.session_state:
            st.session_state["optimization_rmsd"] = load_data(optimization_results_file)

        optimization_rmsd = st.session_state["optimization_rmsd"]

        df_filtered = optimization_rmsd[
            (
                optimization_rmsd.index.get_level_values("calculator")
                == selected_calculator
            )
            & (
                optimization_rmsd.index.get_level_values("settings")
                == selected_settings
            )
        ]

        df_filtered = df_filtered.assign(
            volume_rel=df_filtered.apply(lambda x: x["volume"] / x["volume"][0], axis=1)
        )

        return df_filtered

    df = filter_data(selected_settings, selected_calculator)

    df = df.explode(
        ["step", "rmsd_exp", "rmsd_dft", "volume", "volume_rel", "pot_energy"]
    )

    ## Energy convergence
    st.markdown(f"### Energy convergence")

    fig = px.line(
        df,
        x="step",
        y="pot_energy",
        color="structure",
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## Experimental comparison
    st.markdown(f"### RMSD from experimental structure")

    # def get_rmsd_exp_fig(df, settings, calculator):
    #     if row["rmsd_exp"] is None:
    #         return None
    #     return row["rmsd_exp"]

    fig = px.line(
        df,
        x="step",
        y="rmsd_exp",
        color="structure",
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## DFT-optimized comparison
    st.markdown(f"### RMSD from DFT-optimized structure")

    fig = px.line(
        df,
        x="step",
        y="rmsd_dft",
        color="structure",
        labels={"rmsd_dft": "RMSD from DFT-optimized structure", "step": "Step"},
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## Volume change

    st.markdown(f"### Volume change")

    show_relative_volume = st.toggle("Show relative volume change", False)

    fig = px.line(
        df,
        x="step",
        y="volume_rel" if show_relative_volume else "volume",
        color="structure",
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## Volume vs DFT volume

    st.markdown(f"### Volume comparison with DFT")

    structures = []
    calc_volumes = []
    dft_volumes = []
    for structure in df["structure"].unique():
        df_structure = df[df["structure"] == structure]
        df_structure = df_structure.sort_values("step")
        last_step = df_structure.iloc[-1]
        if last_step["dft_volume"] is not None and not np.isnan(
            last_step["dft_volume"]
        ):
            calc_volumes.append(last_step["volume"])
            dft_volumes.append(last_step["dft_volume"])
            structures.append(structure)

    fig = px.scatter(
        x=dft_volumes,
        y=calc_volumes,
        labels={"x": "DFT volume", "y": "Calculator volume"},
        hover_data={"x": dft_volumes, "y": calc_volumes, "structure": structures},
    )
    offset = 1e3
    fig.add_trace(
        go.Scatter(
            x=[
                min(calc_volumes + dft_volumes) - offset,
                max(calc_volumes + dft_volumes) + offset,
            ],
            y=[
                min(calc_volumes + dft_volumes) - offset,
                max(calc_volumes + dft_volumes) + offset,
            ],
            mode="lines",
            name="y=x",
        )
    )
    fig.update(layout_showlegend=False)

    st.markdown(
        f"**RMSE**: {np.sqrt(np.mean((np.array(calc_volumes) - np.array(dft_volumes))**2)):.2f}"
    )

    st.plotly_chart(fig)
