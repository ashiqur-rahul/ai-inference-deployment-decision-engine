from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests

ELECTRICITY_MAPS_BASE_URL = "https://api.electricitymap.org"

@dataclass
class CarbonSignal:
    zone: str
    carbon_intensity_gco2_kwh: float
    source: str
    raw: Optional[Dict[str, Any]] = None

REGION_TO_ELECTRICITY_MAPS_ZONE = {
    "Germany": "DE",
    "EU Average": "DE",
    "United States": "US-CAL-CISO",
    "Global Average": "DE",
}

STATIC_FALLBACKS = {
    "Germany": 371,
    "EU Average": 213,
    "United States": 373,
    "Global Average": 480,
}

def get_static_carbon_intensity(region: str) -> CarbonSignal:
    return CarbonSignal(region, float(STATIC_FALLBACKS.get(region, STATIC_FALLBACKS["Global Average"])), "static_profile")

def get_electricity_maps_carbon_intensity(zone: str) -> CarbonSignal:
    api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("ELECTRICITY_MAPS_API_KEY is not set.")
    url = f"{ELECTRICITY_MAPS_BASE_URL}/v3/carbon-intensity/latest"
    response = requests.get(url, headers={"auth-token": api_key}, params={"zone": zone}, timeout=15)
    if response.status_code >= 400:
        raise RuntimeError(f"Electricity Maps request failed: {response.status_code} {response.text[:300]}")
    data = response.json()
    value = data.get("carbonIntensity")
    if value is None:
        raise RuntimeError(f"carbonIntensity missing in response: {data}")
    return CarbonSignal(zone, float(value), "electricity_maps_latest", data)

def get_best_available_carbon_signal(region: str) -> CarbonSignal:
    zone = REGION_TO_ELECTRICITY_MAPS_ZONE.get(region, region)
    try:
        return get_electricity_maps_carbon_intensity(zone)
    except Exception:
        return get_static_carbon_intensity(region)

def carbon_adjusted_score_modifier(carbon_intensity_gco2_kwh: float, baseline: float = 400.0) -> float:
    return float(carbon_intensity_gco2_kwh) / max(float(baseline), 1.0)
