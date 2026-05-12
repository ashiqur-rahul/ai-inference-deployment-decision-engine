from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

STATIC_ELECTRICITY_PRICE_USD_KWH = {
    "Germany": 0.32,
    "EU Average": 0.27,
    "United States": 0.14,
    "Global Average": 0.18,
}

@dataclass
class ElectricityPriceSignal:
    region: str
    price_usd_kwh: float
    source: str
    raw: Optional[Dict[str, Any]] = None

def get_static_electricity_price(region: str) -> ElectricityPriceSignal:
    return ElectricityPriceSignal(
        region=region,
        price_usd_kwh=float(STATIC_ELECTRICITY_PRICE_USD_KWH.get(region, STATIC_ELECTRICITY_PRICE_USD_KWH["Global Average"])),
        source="static_profile",
    )

def get_best_available_electricity_price(region: str) -> ElectricityPriceSignal:
    return get_static_electricity_price(region)
