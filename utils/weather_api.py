import requests
from datetime import datetime

def get_weather_forecast(lat, lon, target_date):
    """
    Fetch 7-day weather forecast using Open-Meteo API (no key required).
    Returns: (temp_fahrenheit, rain_probability_percent)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_probability_max",
        "timezone": "America/New_York"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return 80.0, 20.0

    # Find closest matching date
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in data["daily"]["time"]]
    temps = data["daily"]["temperature_2m_max"]
    rains = data["daily"]["precipitation_probability_max"]

    if target_date in dates:
        idx = dates.index(target_date)
        temp_c = temps[idx]
        rain_prob = rains[idx]
        temp_f = round((temp_c * 9 / 5) + 32, 1)
        return temp_f, rain_prob
    else:
        return 80.0, 20.0  # fallback
