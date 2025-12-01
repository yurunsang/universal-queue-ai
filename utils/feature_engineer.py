import pandas as pd
import holidays

def add_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # --- US public holidays ---
    years = df['timestamp'].dt.year.unique()
    us_holidays = holidays.country_holidays('US', years=years)
    df['is_public_holiday'] = df['timestamp'].dt.date.isin(us_holidays).astype(int)

    # --- Approximate school holidays (US pattern) ---
    month = df['timestamp'].dt.month
    df['is_school_holiday'] = 0
    df.loc[month.isin([6,7,8]), 'is_school_holiday'] = 1      # Summer break
    df.loc[month.isin([3,4]), 'is_school_holiday'] = 1        # Spring break
    df.loc[month.isin([12,1]), 'is_school_holiday'] = 1       # Winter break

    return df