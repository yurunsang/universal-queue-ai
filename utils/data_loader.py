import pandas as pd
from utils.feature_engineer import add_temporal_features

def load_wait_data(path="universal_wait_times.csv"):
    """Load and preprocess raw wait time dataset."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_temporal_features(df)
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    return df