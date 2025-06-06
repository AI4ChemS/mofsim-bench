import streamlit as st
import pandas as pd
import os
import numpy as np

from torch import layout
from mof_benchmark import base_dir
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Stability")
st.markdown("# Stability analysis")

# Load the data
stability_file = base_dir / "analysis" / "results" / "stability_results.parquet"
if not os.path.exists(stability_file):
    st.markdown(
        f"File {stability_file} not available. Make sure to run the analysis first."
    )
else:

    # Load the dataframes and store them in session state
    if "stability_df" not in st.session_state:
        st.session_state["stability_df"] = pd.read_parquet(stability_file)

    # Access the dataframes from session state
    stability_df = st.session_state["stability_df"]
    stability_df = stability_df.assign(
        structure=stability_df.index.get_level_values("structure"),
        settings=stability_df.index.get_level_values("settings"),
        calculator=stability_df.index.get_level_values("calculator"),
    )

    selected_settings = st.sidebar.selectbox(
        "Settings",
        stability_df["settings"].unique(),
        format_func=lambda x: str(x).split("/")[-1],
    )

    selected_calculator = st.sidebar.selectbox(
        "Calculator",
        np.sort(
            stability_df[stability_df["settings"] == selected_settings][
                "calculator"
            ].unique()
        ),
    )

    trajectory_df = stability_df[stability_df["calculator"] == selected_calculator]
    trajectory_df = trajectory_df[trajectory_df["settings"] == selected_settings]
    trajectory_df = trajectory_df.explode(
        [
            "rmsd_exp",
            "volume",
            "volume_rel",
            "lengths",
            "angles",
            "step",
            "potential_energy",
            "kinetic_energy",
            "center_of_mass_drift",
            "temperature",
            "stage",
        ]
    )

    coordination_df = stability_df[stability_df["calculator"] == selected_calculator]
    coordination_df = coordination_df[coordination_df["settings"] == selected_settings]
    coordination_df = coordination_df.explode(
        [
            "initial_coordination",
            "final_coordination",
            "symbol",
        ]
    )

    ## Volume change
    st.markdown(f"## Volume change")

    show_relative_volume = st.toggle("Show relative volume change", False)
    stage = st.slider(
        "Stage",
        min_value=0,
        max_value=trajectory_df["stage"].max(),
        value=trajectory_df["stage"].max(),
        key="volume_change_stage",
    )

    fig = px.line(
        trajectory_df[trajectory_df["stage"] == stage],
        x="step",
        y="volume_rel" if show_relative_volume else "volume",
        color="structure",
        labels={
            "volume": "Volume [Å³]",
            "volume_rel": "Relative volume change",
            "step": "NpT Step" if stage >= 2 else "NVT Step",
        },
    )
    fig.update(layout_showlegend=False)

    st.plotly_chart(fig)

    ## Metal coordination change
    st.markdown(f"## Metal coordination change")

    fig = px.scatter(
        coordination_df,
        x="initial_coordination",
        y="final_coordination",
        color="symbol",
        hover_data=["structure"],
        labels={
            "initial_coordination": "Initial average metal coordination",
            "final_coordination": "Final average metal coordination",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[
                min(
                    coordination_df["initial_coordination"].min(),
                    coordination_df["final_coordination"].min(),
                )
                - 0.5,
                max(
                    coordination_df["initial_coordination"].max(),
                    coordination_df["final_coordination"].max(),
                )
                + 0.5,
            ],
            y=[
                min(
                    coordination_df["initial_coordination"].min(),
                    coordination_df["final_coordination"].min(),
                )
                - 0.5,
                max(
                    coordination_df["initial_coordination"].max(),
                    coordination_df["final_coordination"].max(),
                )
                + 0.5,
            ],
            mode="lines",
            line=dict(color="blue"),
            showlegend=False,
            opacity=0.2,
        )
    )
    st.plotly_chart(fig)

    ## Volume change start to finish
    st.markdown(f"## Volume change relative")

    final_volume_df = trajectory_df[
        (trajectory_df["stage"] >= stage)
        & (
            trajectory_df["step"]
            >= trajectory_df[trajectory_df["stage"] >= stage]["step"].max()
            - 1  # TODO: remove -1 when data has been recalculated (with last step 19999, not 19998)
        )
    ]
    final_volume_df.loc[:, "volume_rel"] = final_volume_df.loc[:, "volume_rel"] - 1

    fig = px.histogram(
        final_volume_df,
        x="volume_rel",
        color="structure",
        nbins=50,
        labels={"volume_rel": "Relative volume change"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    ## Center of mass drift
    st.markdown(f"## Center of mass drift")

    fig = px.line(
        trajectory_df[trajectory_df["stage"] >= stage],
        x="step",
        y="center_of_mass_drift",
        color="structure",
        labels={
            "center_of_mass_drift": "Center of mass drift [Å]",
            "step": "NpT Step",
        },
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## Potential energy drift

    st.markdown(f"## Potential energy drift")

    fig = px.line(
        trajectory_df[trajectory_df["stage"] >= stage],
        x="step",
        y="potential_energy",
        color="structure",
        labels={"potential_energy": "Potential energy [eV]", "step": "NpT Step"},
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    ## Trajectory temperature and volume
    st.markdown(f"## Trajectory temperature")

    stage = st.slider(
        "Stage",
        min_value=0,
        max_value=trajectory_df["stage"].max(),
        value=trajectory_df["stage"].max(),
    )
    structure = st.selectbox(
        "Structure",
        trajectory_df[trajectory_df["stage"] == stage]["structure"].unique(),
    )

    fig = px.line(
        trajectory_df[
            (trajectory_df["stage"] == stage)
            & (trajectory_df["structure"] == structure)
        ],
        x="step",
        y="volume",
        color="structure",
        labels={"volume": "Volume [$\\AA^3$]", "step": "NpT Step"},
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)

    fig = px.line(
        trajectory_df[
            (trajectory_df["stage"] == stage)
            & (trajectory_df["structure"] == structure)
        ],
        x="step",
        y="temperature",
        color="structure",
        labels={"temperature": "Temperature [K]", "step": "NpT Step"},
    )
    fig.update(layout_showlegend=False)
    st.plotly_chart(fig)
