import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.set_page_config(page_title="Overview")

st.title("MOFBench")

st.markdown(
    "Welcome to MOFBench, a comprehensive benchmark suite for evaluating the performance of Machine Learning Interatomic Potentials (MLIPs) on Metal-Organic Framework (MOF) properties. "
    "This platform provides a standardized collection of benchmarks based on state-of-the-art MLIPs and diverse MOF structures sourced from prominent databases including CoRE-MOF, QMOF, IZA, and Curated-COF. "
    "Explore the sections below to delve into the details of each benchmark."
)


st.markdown(
    """Please select a task from the sidebar to view detailed results and analyses."""
)
