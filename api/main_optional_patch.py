try:
    from prometheus_client import make_asgi_app
except Exception:
    make_asgi_app = None

from telemetry import record_recommendation

def mount_metrics_if_available(app):
    if make_asgi_app is not None:
        app.mount("/metrics", make_asgi_app())
    return app
