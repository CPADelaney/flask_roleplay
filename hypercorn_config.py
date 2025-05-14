# hypercorn_config.py
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
worker_class = "asyncio"
workers = 1
startup_timeout = 60  # Increase to 60 seconds (or more if needed for debugging)
# graceful_timeout = 5 # Default is 3 seconds, can adjust if needed for shutdown

# Optional: Logging configuration (Hypercorn's own logs)
# loglevel = "info" # Can be debug, info, warning, error, critical
# error_logfile = "-"  # Log errors to stderr
# access_logfile = "-" # Log access to stdout
# access_logformat = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s' # Example Apache combined format
