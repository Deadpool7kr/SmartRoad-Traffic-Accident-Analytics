from __future__ import annotations

from typing import Any, Dict
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

UK_CITIES = {
    "London": (51.5074, -0.1278),
    "Birmingham": (52.4862, -1.8904),
    "Manchester": (53.4808, -2.2426),
    "Leeds": (53.8008, -1.5491),
    "Liverpool": (53.4084, -2.9916),
    "Bristol": (51.4545, -2.5879),
    "Glasgow": (55.8642, -4.2518),
    "Edinburgh": (55.9533, -3.1883),
    "Cardiff": (51.4816, -3.1791),
    "Nottingham": (52.9548, -1.1581),
}

WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def fetch_current_weather(city: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch current weather for a supported UK city from Open-Meteo."""
    if city not in UK_CITIES:
        raise ValueError(f"Unsupported city: {city}")

    latitude, longitude = UK_CITIES[city]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,precipitation,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    current = payload.get("current")
    if not isinstance(current, dict):
        raise ValueError("Weather API returned no current conditions")

    current["weather_description"] = WEATHER_CODE_LABELS.get(
        int(current.get("weather_code", -1)), "Unknown"
    )
    current["city"] = city
    current["latitude"] = latitude
    current["longitude"] = longitude
    return current
