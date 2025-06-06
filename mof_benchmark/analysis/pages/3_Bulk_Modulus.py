import streamlit as st
import pandas as pd
import os
import yaml

import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
from ase.eos import EquationOfState
from ase import units
import ast


from mof_benchmark import base_dir

st.set_page_config(page_title="Bulk Modulus")
st.markdown("# Bulk modulus analysis")

# Load the data
bulk_modulus_file = base_dir / "analysis" / "results" / "bulk_modulus_results.parquet"
if not os.path.exists(bulk_modulus_file):
    st.markdown(
        f"File {bulk_modulus_file} not available. Make sure to run the analysis first."
    )

else:

    bulk_modulus_results = pd.read_parquet(bulk_modulus_file)

    settings_col = bulk_modulus_results.index.get_level_values("settings")

    selected_settings = st.sidebar.selectbox(
        "Settings",
        settings_col.unique(),
        format_func=lambda x: x.split("/")[-1],
    )

    calculator_col = bulk_modulus_results[
        settings_col == selected_settings
    ].index.get_level_values("calculator")

    selected_calculator = st.sidebar.selectbox(
        "Calculator",
        np.sort(calculator_col.unique()),
    )

    bulk_modulus_df = bulk_modulus_results[
        (
            bulk_modulus_results.index.get_level_values("calculator")
            == selected_calculator
        )
        & (settings_col == selected_settings)
    ]
    bulk_modulus_df = bulk_modulus_df.assign(
        structure=bulk_modulus_df.index.get_level_values("structure")
    )

    ## Bulk modulus parallel plot
    st.markdown("### Bulk modulus prediction vs DFT")

    bulk_modulus_df_valid = bulk_modulus_df.dropna(subset=["B", "dft_B"])
    rmse = np.sqrt(
        ((bulk_modulus_df_valid["B"] - bulk_modulus_df_valid["dft_B"]) ** 2).mean()
    )
    st.markdown(
        f"Computed structures: {len(bulk_modulus_df)}, B NaNs: {bulk_modulus_df['B'].isnull().sum()}, DFT B NaNs: {bulk_modulus_df['dft_B'].isnull().sum()}"
    )
    st.markdown(f"RMSE: {rmse:.2f} ({len(bulk_modulus_df_valid)} structures)")

    failed_structures = bulk_modulus_df[
        (bulk_modulus_df["B"].isnull()) & ~(bulk_modulus_df["dft_B"].isnull())
    ]
    if not failed_structures.empty:
        with st.expander(f"Failed structures ({len(failed_structures)})"):
            st.table(failed_structures["structure"].values)

    fig = px.scatter(
        bulk_modulus_df,
        x="dft_B",
        y="B",
        hover_data=["structure"],
        labels={
            "B": "Bulk modulus prediction (GPa)",
            "dft_B": "Bulk modulus DFT (GPa)",
        },
    )
    min_B = min(bulk_modulus_df["B"].min(), bulk_modulus_df["dft_B"].min())
    max_B = max(bulk_modulus_df["B"].max(), bulk_modulus_df["dft_B"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_B - 10, max_B + 10],
            y=[min_B - 10, max_B + 10],
            mode="lines",
            name="y=x",
        )
    )
    fig.update(
        layout_showlegend=False,
    )
    # fig.update_xaxes(
    #     range=[
    #         max(bulk_modulus_df["B"].min() - 10, 0),
    #         min(bulk_modulus_df["B"].max() + 10, 80),
    #     ]
    # )
    # fig.update_yaxes(
    #     range=[
    #         max(bulk_modulus_df["B"].min() - 10, 0),
    #         min(bulk_modulus_df["B"].max() + 10, 80),
    #     ]
    # )
    st.plotly_chart(fig)

    ## Bulk modulus fits
    st.markdown("### Bulk modulus EOS")

    selected_structure = st.selectbox(
        "Select structure", np.sort(bulk_modulus_df["structure"].unique())
    )

    show_eos_fit = st.checkbox("Show EOS fit", value=True)

    structure_df = bulk_modulus_df[
        bulk_modulus_df["structure"] == selected_structure
    ].iloc[0]

    fig = px.scatter(
        x=structure_df["volumes"],
        y=structure_df["energies"],
        labels={"volumes": "Volume (A^3)", "energies": "Energy (eV)"},
    )
    if show_eos_fit:
        try:
            eos = EquationOfState(
                structure_df["volumes"],
                structure_df["energies"],
                eos="birchmurnaghan",
            )
            v0, e0, B = eos.fit(warn=False)
            plot_data = eos.getplotdata()
            fig.add_trace(
                go.Scatter(
                    x=plot_data[-4],
                    y=plot_data[-3],
                    mode="lines",
                    name="MLIP Murnaghan fit",
                )
            )
            st.markdown(f"V0: {v0:.2f} A^3")
            st.markdown(f"E0: {e0:.2f} eV")
            st.markdown(f"B: {B / units.kJ * 1.0e24 :.2f} GPa")
        except Exception as e:
            st.error(f"Error fitting EOS: {e}")

    st.plotly_chart(fig)

    fig = px.scatter(
        x=np.array(ast.literal_eval(structure_df["dft_vol"]), dtype=float),
        y=np.array(ast.literal_eval(structure_df["dft_eng"]), dtype=float)
        * units.Hartree,
        labels={"volumes": "Volume (A^3)", "energies": "Energy (eV)"},
    )

    if show_eos_fit:
        try:
            eos = EquationOfState(
                np.array(ast.literal_eval(structure_df["dft_vol"]), dtype=float),
                np.array(ast.literal_eval(structure_df["dft_eng"]), dtype=float)
                * units.Hartree,
                eos="birchmurnaghan",
            )
            v0, e0, B = eos.fit(warn=False)
            plot_data = eos.getplotdata()
            fig.add_trace(
                go.Scatter(
                    x=plot_data[-4],
                    y=plot_data[-3],
                    mode="lines",
                    name="DFT Murnaghan fit",
                )
            )
            st.markdown(f"DFT V0: {v0:.2f} A^3")
            st.markdown(f"DFT E0: {e0:.2f} eV")
            st.markdown(f"DFT B: {B / units.kJ * 1.0e24 :.2f} GPa")
        except Exception as e:
            st.error(f"Error fitting EOS: {e}")

    st.plotly_chart(fig)
