# Prometheus and Grafana Telemetry

This project includes a monitoring scaffold:

```text
deployment/monitoring/
```

## Run

```bash
cd deployment/monitoring
docker compose up
```

Services:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

The optional telemetry module can expose metrics such as:

- recommendation request count
- selected latency
- selected carbon estimate
- selected monthly cost
