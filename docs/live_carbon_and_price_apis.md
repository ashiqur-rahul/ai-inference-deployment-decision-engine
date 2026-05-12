# Live Carbon and Electricity Price APIs

## Carbon Signals

The project includes `src/cloud_carbon.py`.

It supports:

- static carbon intensity fallback
- optional Electricity Maps API integration

Environment variable:

```bash
ELECTRICITY_MAPS_API_KEY=your_key_here
```

If no API key is present, the system falls back to local static profiles.

## Electricity Price Signals

The project includes `src/electricity_price.py`.

At the moment, it uses static regional electricity prices because live price APIs differ by country, market, and provider.

Future extensions can connect:

- day-ahead market APIs
- supplier tariff APIs
- cloud region pricing APIs
- custom CSV price feeds
