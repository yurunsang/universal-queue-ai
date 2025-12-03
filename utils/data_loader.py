import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # cache for 1 hour to avoid hitting GitHub too often
def load_wait_data():
    """
    Load the latest Universal Studios wait time data directly from GitHub.
    Auto-refreshes every hour.
    """
    url = "https://raw.githubusercontent.com/yurunsang/universal-wait-logger/main/universal_wait_times.csv"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Failed to load data from GitHub: {e}")
        return pd.DataFrame()

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Add a convenience 'date' column
    if "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.date

    return df