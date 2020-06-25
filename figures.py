import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from calculations import calc_prob_between
from mpmath import betainc


def plot_histogram(df, handle=go.Figure()):
    handle.data = []
    handle.layout = {}
    handle = px.histogram(df, x="outcome", color="group")
    handle.update_layout(barmode="overlay")
    handle.update_traces(opacity=0.5)
    return handle


def plot_bar(df, handle=go.Figure()):
    handle.data = []
    handle.layout = {}
    handle = go.Figure(
        data=[
            go.Bar(name="control", x=df["x"], y=df["control"]),
            go.Bar(name="treatment", x=df["x"], y=df["treatment"]),
        ]
    )
    handle.update_layout(barmode="overlay")
    handle.update_traces(opacity=0.9)
    return handle


def plot_pdf(df, rate_control, rate_treatment, handle=go.Figure()):
    handle.data = []
    handle.layout = {}
    handle = go.Figure(
        data=[
            go.Scatter(name="control", x=df["x"], y=df["control"]),
            go.Scatter(name="treatment", x=df["x"], y=df["treatment"]),
        ]
    )
    handle.add_shape(
        # Line reference to the axes
        go.layout.Shape(
            type="line",
            xref="x",
            yref="paper",
            x0=rate_control,
            y0=0,
            x1=rate_control,
            y1=1,
            line=dict(color="Black", width=1, dash="dash"),
        )
    )
    handle.add_shape(
        # Line reference to the axes
        go.layout.Shape(
            type="line",
            xref="x",
            yref="paper",
            x0=rate_treatment,
            y0=0,
            x1=rate_treatment,
            y1=1,
            line=dict(color="Black", width=1, dash="dash"),
        )
    )

    handle.update_layout(
        title="Conversion Rates Probability Density Function",
        xaxis_title="Rate",
        yaxis_title="Probability Density",
    )

    return handle


def calc_beta_mode(a, b):
    """this function calculate the mode (peak) of the Beta distribution"""
    return (a - 1) / (a + b - 2)


def ab_plot(betas, range, names, handle=go.Figure()):
    """this function plots the Beta distribution"""
    x = np.linspace(range[0], range[1], 1000)
    handle.data = []
    handle.layout = {}

    for f, name in zip(betas, names):
        y = f.pdf(
            x
        )  # this for calculate the value for the PDF at the specified x-points
        y_mode = calc_beta_mode(f.args[0], f.args[1])
        y_var = f.var()  # the variance of the Beta distribution
        handle.add_trace(go.Scatter(name=name, x=x, y=y),)

    return handle


def print_report(beta_C, beta_T):

    larger = None
    smaller = None
    lift = None
    prob = None

    if (beta_T.mean() > beta_C.mean()):
        lift = (beta_T.mean() - beta_C.mean()) / beta_C.mean()
        larger = "Treatment"
        smaller = "Control"
        prob = calc_prob_between(beta_C, beta_T)
    else:
        lift = (beta_C.mean() - beta_T.mean()) / beta_C.mean()
        larger = "Control"
        smaller = "Treatment"
        prob = calc_prob_between(beta_C, beta_T)

    # calculating the probability for Test to be better than Control
    # print(beta_T.mean(), beta_C.mean(), lift)
    # print(larger, smaller)
    control = beta_C.ppf([0.025, 0.5, 0.975])
    treatment = beta_T.ppf([0.025, 0.5, 0.975])

    precision = 3
    results = {
        "result": f"{larger} outperforms {smaller} by {lift*100:2.2f}% with {prob*100:2.1f}% probability.",
        "treatment (lower 95% c.i., upper 95% c.i.)": str(
            round(treatment[1], precision)
        )
        + " ("
        + str(round(treatment[0], precision))
        + ", "
        + str(round(treatment[2], precision))
        + ")",
        "control (lower 95% c.i., upper 95% c.i.)": str(round(control[1], precision))
        + " ("
        + str(round(control[0], precision))
        + ", "
        + str(round(control[2], precision))
        + ")",
    }

    st.write(results)
