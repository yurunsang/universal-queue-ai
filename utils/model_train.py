import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from utils.feature_engineer import add_temporal_features

def train_per_ride_models(
    csv_url="https://raw.githubusercontent.com/yurunsang/universal-wait-logger/main/universal_wait_times.csv",
    output_dir="model_result"
):
    """
    Train a separate RandomForest model for each ride and save them individually.
    The data is loaded directly from GitHub (auto-updated by GitHub Actions).
    """
    print(f"🌐 Loading data from GitHub: {csv_url}")
    df = pd.read_csv(csv_url)
    df = add_temporal_features(df)

    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [
        'day_of_week', 'hour', 'is_weekend',
        'temp_f', 'rain_probability', 'wind_speed_mph',
        'is_public_holiday', 'is_school_holiday'
    ]
    target = 'wait_time'

    all_rides = sorted(df['ride'].unique())
    print(f"🎢 Training {len(all_rides)} ride-specific models...\n")

    for ride in all_rides:
        df_ride = df[(df['ride'] == ride) & (df['is_open'])].copy()
        if len(df_ride) < 30:
            print(f"⚠️ Skipping {ride}: insufficient data ({len(df_ride)} samples)")
            continue

        X = df_ride[feature_cols]
        y = df_ride[target]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        model_path = os.path.join(output_dir, f"{ride.replace('/', '_')}.pkl")
        joblib.dump(model, model_path)

        print(f"✅ {ride[:45]:45s} — saved ({len(df_ride)} samples, R²={score:.2f})")

    print("\n📦 All ride-specific models saved to:", output_dir)


if __name__ == "__main__":
    train_per_ride_models()