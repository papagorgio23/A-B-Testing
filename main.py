import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as scs
import datetime
import os
import sys
import math
from scipy.stats import beta
from datasets import get_datasets, load_data
from figures import plot_histogram, plot_bar, plot_pdf, ab_plot, print_report
from calculations import min_sample_size, sample_power_probtest


state = {
    "plan_expt": False,
    "data_selected": False,
    "save": False,
}
selected_dataset = " "

# DATE_COLUMN = "date/time"
DATA_DIR = "./Data/"
STATIC_URL = (
    "./"
)
logo = STATIC_URL + "freedom_logo.png"

# Title
st.title("A/B Testing Framework")

# Header
st.header("**Power Analysis**")

# Subheader
st.subheader("_Adjust the inputs in the sidebar and calculate to see results below_")

st.text("""Assumptions:
- Independent Groups
- Dichotomous Result (Yes/No)
- 2 Tailed Test (Treatment Group can be better or worse than control group)
""")

# Logo
st.sidebar.image(logo, use_column_width=True)

# Plan your experiment
#st.sidebar.subheader("Adjust the inputs and calculate to see results:")
bcr = float(st.sidebar.text_input("Base Conversion Rate", value=0.2))
mde = float(st.sidebar.text_input("Min detectable effect", value=0.05))
power = float(st.sidebar.text_input("Power (80 Baseline)", value=0.8))
sig_level = float(st.sidebar.text_input("Significance Level", value=0.05))
weekly_samples = int(st.sidebar.text_input("Weekly Samples", value=4000))
state["plan_expt"] = st.sidebar.button("Calculate Sample Size")

if state["plan_expt"]:
    sample_size = sample_power_probtest(
        bcr, bcr*(1+mde), power=power, sig=sig_level)
    new_conversion_rate = bcr*(1+mde)
    num_of_weeks = math.ceil(sample_size*2/weekly_samples)

    sample_size_print = """Sample Size per Group = {:,}  

    Baseline Conversion Rate = {:.1%}  
    Target Conversion Rate = {:.1%}  
      
    Samples per Week = {}  
    Number of Weeks = {}  
    """.format(sample_size, bcr, new_conversion_rate, weekly_samples, num_of_weeks)
    if sample_size > 0:
        st.success(sample_size_print)
    if sample_size < 1:
        st.error(sample_size_print)

# Select the dataset to analyze
datasets = get_datasets(DATA_DIR)
#st.sidebar.subheader("Data Sources")
#selected_dataset = st.sidebar.selectbox("", datasets)
data_url = DATA_DIR + selected_dataset


# Load and display selected dataset
df = pd.DataFrame()
status_text = st.empty()
if selected_dataset != " ":
    status_text.text("Loading data...")
    df = load_data(data_url)
    df.index.freq = pd.infer_freq(df.index, warn=True)
    st.header("AB Test")
    st.subheader("Raw data")
    st.dataframe(df)
    state["data"] = True
    status_text.text("Loading data...Done!")
    hist_chart = st.empty()

    # Plot histogram
    st.subheader("Histogram for treatment and control ")
    hist_chart = st.empty()
    fig = plot_histogram(df)
    hist_chart.plotly_chart(fig)

    df.pivot_table(values="outcome", index="group", aggfunc=np.sum)
    ab_summary = df.pivot_table(
        values="outcome", index="group", aggfunc=np.sum)

    # Get rates
    ab_summary["total"] = df.pivot_table(
        values="outcome", index="group", aggfunc=lambda x: len(x)
    )
    ab_summary["rate"] = df.pivot_table(values="outcome", index="group")
    st.subheader("Outcome rate for treatment and control ")
    st.dataframe(ab_summary)

    control_total = ab_summary.loc["control"]["total"]
    control_outcome = ab_summary.loc["control"]["outcome"]
    treatment_total = ab_summary.loc["treatment"]["total"]
    treatment_outcome = ab_summary.loc["treatment"]["outcome"]

    avg_rate = (
        ab_summary.loc["control"]["rate"] + ab_summary.loc["treatment"]["rate"]
    ) * 0.5

    pct = 0.2
    lower_rate = avg_rate * (1 - pct)
    upper_rate = avg_rate * (1 + pct)

    # here we create the Beta functions for the two sets
    a_C, b_C = control_outcome + 1, control_total - control_outcome + 1
    beta_C = beta(a_C, b_C)
    a_T, b_T = treatment_outcome + 1, treatment_total - treatment_outcome + 1
    beta_T = beta(a_T, b_T)

    st.subheader("AB Test Results")
    st.subheader("Probability Density for Control and Treatment")
    prob_chart = st.empty()
    ab_fig = ab_plot(
        [beta_C, beta_T], [lower_rate, upper_rate], names=["Control", "Treatment"]
    )
    prob_chart.plotly_chart(ab_fig)

    st.subheader("Explanation")
    print_report(beta_C, beta_T)
