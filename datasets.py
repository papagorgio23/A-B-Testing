import os
import streamlit as st
import pandas as pd


def get_datasets(url) -> list:
    all_files = os.listdir(url)
    csv_files = [file for file in all_files if file.endswith(".csv")]
    csv_files = [" "] + csv_files
    return csv_files


@st.cache
def load_data(url) -> pd.DataFrame:
    data = pd.read_csv(url)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data.set_index("timestamp", inplace=True)
    return data
