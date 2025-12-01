import streamlit as st
import pandas as pd
import os, joblib
from datetime import date, datetime, timedelta
import holidays
import plotly.express as px

# ===== Custom Utilities =====
from utils.data_loader import load_wait_data
from utils.weather_api import get_weather_forecast

# ==========================================================
# 1. Page setup & style
# ==========================================================
st.set_page_config(page_title="🎢 Universal Queue Optimizer - Florida", layout="wide")

# --- Custom CSS for banner, leaderboard, and sidebar footer ---
st.markdown(
    """
    <style>
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 0rem;}
        .banner-container {
            position: relative;
            width: 100%;
            overflow: hidden;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }
        .banner-container img {
            width: 100%;
            height: 300px;
            object-fit: cover;
        }
        .banner-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.35);
        }
        .banner-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 1.6rem;
            font-weight: 600;
            text-align: center;
            text-shadow: 0 0 10px rgba(0,0,0,0.8);
            animation: fadeIn 1.8s ease-in-out;
            z-index: 2;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        /* 🏆 Glass Leaderboard */
        .leaderboard {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .glass-card {
            flex: 1;
            padding: 1rem 1.3rem;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 6px 25px rgba(0,0,0,0.15);
            color: #111;
        }
        .glass-card h4 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 1.1rem;
            color: #1e1e1e;
            font-weight: 700;
        }
        .ride-item {
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
            color: #111;
        }
        .ride-item b {
            color: #000;
        }
        .ride-item span {
            color: #333;
            opacity: 0.85;
        }
        .ride-item:hover {
            transform: translateX(4px);
            transition: 0.25s;
        }

        /* Sidebar footer */
        section[data-testid="stSidebar"] div.block-container {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .sidebar-footer {
            margin-top: auto;
            font-size: 0.9rem;
            opacity: 0.85;
            padding-top: 1rem;
        }
    </style>

    <div class="banner-container">
        <img src="https://images.unsplash.com/photo-1569789010436-421d71a9fc38?q=80&w=1725&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D">
        <div class="banner-overlay"></div>
        <div class="banner-text">Skip the lines, feel the thrill — your AI-powered Universal day planner.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🎢 Universal Queue Optimizer - Florida")

MODEL_DIR = "model_result"

# ==========================================================
# 2. Load dataset
# ==========================================================
@st.cache_data
def get_data():
    return load_wait_data()

df = get_data()

# ==========================================================
# 3. Sidebar controls
# ==========================================================
st.sidebar.header("🔧 Trip Settings")

selected_park = st.sidebar.selectbox("Select Park", df["park"].unique())
selected_date = st.sidebar.date_input("Select Visit Date", min_value=df["timestamp"].min().date())

def available_ride_models():
    if not os.path.exists(MODEL_DIR):
        return []
    model_files = [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    all_rides = sorted(df[df["park"] == selected_park]["ride"].unique())
    return [r for r in all_rides if r.replace("/", "_") in model_files]

valid_rides = available_ride_models()
select_all = st.sidebar.checkbox("✅ Select All Rides with Models", value=False)
rides = valid_rides if select_all else st.sidebar.multiselect("🎢 Select Rides You Plan to Visit", valid_rides)

if not valid_rides:
    st.warning("⚠️ No trained models found in `model_result/`. Please run `python -m utils.model_train` first.")

st.sidebar.markdown("---")
st.sidebar.caption("Model version: per-ride RandomForest models")

# ==========================================================
# 4. Predictive Engine
# ==========================================================
st.subheader("🤖 Predict & Plan Your Day")

lat, lon = 28.4745, -81.4717
temp_f, rain_prob = get_weather_forecast(lat, lon, selected_date)
st.info(f"🌤 Forecast for {selected_date}: **{temp_f}°F**, 🌧️ Rain Probability: **{rain_prob}%**")

def infer_holiday_flags(selected_date):
    us_holidays = holidays.country_holidays('US', subdiv='FL', years=[selected_date.year])
    is_public = int(selected_date in us_holidays)
    is_school = int(selected_date.month in [3,4,6,7,8,12,1])
    return is_public, is_school

is_public, is_school = infer_holiday_flags(selected_date)
is_weekend = 1 if selected_date.weekday() >= 5 else 0

def load_ride_model(ride):
    model_path = os.path.join(MODEL_DIR, f"{ride.replace('/', '_')}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pred_results = []
for ride in rides:
    model = load_ride_model(ride)
    if model is None:
        continue
    X_pred = pd.DataFrame([{
        "day_of_week": selected_date.weekday(),
        "hour": 12,
        "is_weekend": is_weekend,
        "temp_f": temp_f,
        "rain_probability": rain_prob,
        "wind_speed_mph": 5,
        "is_public_holiday": is_public,
        "is_school_holiday": is_school
    }])
    wait_pred = int(round(model.predict(X_pred)[0]))
    pred_results.append({"ride": ride, "predicted_wait": wait_pred})

# ==========================================================
# 5. Visual Insights (Glass Leaderboard)
# ==========================================================
if len(pred_results) > 0:
    pred_df = pd.DataFrame(pred_results)
    optimized_route = pred_df.sort_values("predicted_wait").reset_index(drop=True)
    optimized_route["order"] = optimized_route.index + 1

    # 🏆 Glass-style leaderboard at top
    fastest = optimized_route.nsmallest(5, "predicted_wait")
    slowest = optimized_route.nlargest(5, "predicted_wait")

    st.markdown("### 🏆 Quick Summary")
    st.markdown(
        f"""
        <div class="leaderboard">
            <div class="glass-card">
                <h4>⚡ Fastest Rides</h4>
                {''.join([f"<div class='ride-item'>🏎️ <b>{r['ride']}</b> <span>— {r['predicted_wait']} min</span></div>" for _, r in fastest.iterrows()])}
            </div>
            <div class="glass-card">
                <h4>🐢 Longest Queues</h4>
                {''.join([f"<div class='ride-item'>🎢 <b>{r['ride']}</b> <span>— {r['predicted_wait']} min</span></div>" for _, r in slowest.iterrows()])}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # 📊 Predicted Wait Times
    # ------------------------------------------------------
    st.subheader("📊 Predicted Wait Times by Ride")
    fig_bar = px.bar(
        optimized_route,
        x="predicted_wait",
        y="ride",
        orientation="h",
        color="predicted_wait",
        color_continuous_scale="RdYlBu_r",
        labels={"predicted_wait": "Predicted Wait (min)", "ride": "Ride"},
        title="Predicted Queue Duration per Ride"
    )
    fig_bar.update_layout(height=600, xaxis_title="Wait (minutes)", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------
    # 🕓 Timeline
    # ------------------------------------------------------
    st.subheader("🕓 Recommended Day Schedule")
    start_time = datetime.combine(selected_date, datetime.strptime("09:00", "%H:%M").time())
    total_time = 0
    schedule = []
    for _, row in optimized_route.iterrows():
        start = start_time + timedelta(minutes=total_time)
        end = start + timedelta(minutes=row["predicted_wait"] + 10)
        total_time += row["predicted_wait"] + 10
        schedule.append({
            "ride": row["ride"],
            "start_time": start,
            "end_time": end,
            "predicted_wait": row["predicted_wait"]
        })
    schedule_df = pd.DataFrame(schedule)
    fig_timeline = px.timeline(
        schedule_df,
        x_start="start_time",
        x_end="end_time",
        y="ride",
        color="predicted_wait",
        color_continuous_scale="RdYlBu_r",
        title="⏰ Your Optimized Ride Plan"
    )
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(height=600)
    st.plotly_chart(fig_timeline, use_container_width=True)

    with st.expander("📋 Show Detailed Predictions"):
        st.dataframe(
            optimized_route.rename(columns={
                "order": "🎯 Order",
                "ride": "🎢 Ride",
                "predicted_wait": "⏱ Predicted Wait (min)"
            }),
            use_container_width=True
        )
else:
    st.warning("Please select at least one ride that has a model to generate predictions.")

# ==========================================================
# 6. Sidebar Footer
# ==========================================================
df_park = df[df["park"] == selected_park]
avg_wait = df_park.groupby("date")["wait_time"].mean().mean()

if temp_f > 90 or is_public or is_school:
    crowd_level = "🚨 Peak Crowd Expected"
elif avg_wait > 40:
    crowd_level = "⚠️ Busy Day"
else:
    crowd_level = "✅ Normal Day"

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="sidebar-footer">
        <h4>📈 Crowd Level Indicator</h4>
        <p><b>{crowd_level}</b><br>
        based on historical averages, weather, and holiday patterns.</p>
        <br>
        <p>© 2025 Universal Queue Optimizer - Florida<br>
        smart analytics edition 🚀</p>
        <hr style="margin:0.8rem 0; border:0; border-top:1px solid rgba(255,255,255,0.3);">
        <p style="font-size:0.85rem; opacity:0.8;">
            🧠 Designed & Developed by <b>Runsang&nbsp;Yu</b><br>
            📧 <a href="mailto:yurunsang19@gmail.com" target="_blank" style="color:#58a6ff;">yurunsang19@gmail.com</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
