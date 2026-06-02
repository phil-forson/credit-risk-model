"""WSGI entrypoint when Render/Gunicorn is started as ``gunicorn app:app``."""
from server import app  # noqa: F401
