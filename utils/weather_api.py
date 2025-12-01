import requests

def get_weather_forecast(lat, lon, date):
    """
    Fetch forecast for a specific date using Open-Meteo API.
    Returns (temp_f, rain_probability).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,precipitation_probability_max"
        f"&timezone=America/New_York&start_date={date}&end_date={date}"
    )

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        temp_c = data["daily"]["temperature_2m_max"][0]
        rain_prob = data["daily"]["precipitation_probability_max"][0]
        temp_f = temp_c * 9/5 + 32
        return round(temp_f, 1), round(rain_prob, 1)
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return 80.0, 20.0  # default fallback