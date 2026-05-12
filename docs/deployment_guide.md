# Deployment Guide

## Local Dashboard

```bash
streamlit run app/app.py
```

## Local API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build:

```bash
docker build -t ai-deployment-decision-engine .
```

Run dashboard:

```bash
docker run -p 8501:8501 ai-deployment-decision-engine
```

## Docker Compose

```bash
docker compose up --build
```

Dashboard:

```text
http://localhost:8501
```

API:

```text
http://localhost:8000/docs
```

## Notes

For a public demo, deploy the dashboard and API separately or use a platform that supports multiple services.
