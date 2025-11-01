# config_startup.py
from dotenv import load_dotenv
load_dotenv()                      # dev only; no‑op in Render

import os
from agents import set_default_openai_key, set_default_openai_api

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing")

set_default_openai_key(api_key)    # <— global for the SDK
set_default_openai_api("chat_responses")   # optional

# Ensure Celery broker/backends default to Redis if not specified so workers
# inherit sane defaults in development environments.
os.environ.setdefault("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
os.environ.setdefault(
    "CELERY_RESULT_BACKEND",
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)
