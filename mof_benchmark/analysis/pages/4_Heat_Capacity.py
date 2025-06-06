import streamlit as st
import pandas as pd
import os

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from mof_benchmark import base_dir

st.set_page_config(page_title="Heat Capacity")
st.markdown("# Heat capacity analysis")

# Load the data
heat_capacity_file = base_dir / "analysis" / "results" / "heat_capacity_results.parquet"
if not os.path.exists(heat_capacity_file):
    st.markdown(
        f"File {heat_capacity_file} not available. Make sure to run the analysis first."
    )

else:

    heat_capacity_results = pd.read_parquet(heat_capacity_file)
    heat_capacity_results = heat_capacity_results.assign(
        structure=heat_capacity_results.index.get_level_values("structure")
    )
    heat_capacity_results = heat_capacity_results.assign(
        settings=heat_capacity_results.index.get_level_values("settings")
    )
    heat_capacity_results = heat_capacity_results.assign(
        calculator=heat_capacity_results.index.get_level_values("calculator")
    )

    settings_col = heat_capacity_results.index.get_level_values("settings")

    selected_settings = st.sidebar.selectbox(
        "Settings",
        settings_col.unique(),
        format_func=lambda x: x.split("/")[-1],
    )
    selected_calculator = st.sidebar.selectbox(
        "Calculator",
        np.sort(
            heat_capacity_results[settings_col == selected_settings][
                "calculator"
            ].unique()
        ),
    )

    heat_capacity_df = heat_capacity_results[
        (heat_capacity_results["calculator"] == selected_calculator)
        & (settings_col == selected_settings)
    ]
    heat_capacity_df = heat_capacity_df.assign(structure=heat_capacity_df["structure"])

    ## Heat capacity parallel plot
    st.markdown("### Heat capacity prediction vs DFT")

    temp_300_index = np.where(heat_capacity_df.iloc[0]["temperatures"] == 300)[0].item()
    cv_300 = heat_capacity_df["heat_capacity_g"].apply(lambda x: x[temp_300_index])

    rmse = np.sqrt(((cv_300 - heat_capacity_df["dft_cv"]) ** 2).mean())
    st.markdown(f"RMSE: {rmse:.3f} ({len(heat_capacity_df)} structures)")
    failed_structures = cv_300.isnull().sum()
    st.markdown(f"Failed structures: {failed_structures}")

    print(heat_capacity_df["dft_cv"])

    fig = px.scatter(
        heat_capacity_df,
        x="dft_cv",
        y=cv_300.values,
        hover_data=["structure"],
        labels={
            "y": "Heat capacity prediction (J/g/K)",
            "dft_cv": "Heat capacity DFT (J/g/K)",
        },
    )
    min_cv = min(heat_capacity_df["dft_cv"].min(), cv_300.min())
    max_cv = max(heat_capacity_df["dft_cv"].max(), cv_300.max())
    fig.add_trace(
        go.Scatter(
            x=[min_cv - 0.1, max_cv + 0.1],
            y=[min_cv - 0.1, max_cv + 0.1],
            mode="lines",
            name="y=x",
        )
    )
    fig.update(
        layout_showlegend=False,
        layout_xaxis_range=[
            heat_capacity_df["dft_cv"].min() - 0.1,
            heat_capacity_df["dft_cv"].max() + 0.1,
        ],
        layout_yaxis_range=[cv_300.min() - 0.1, cv_300.max() + 0.1],
    )
    st.plotly_chart(fig)
