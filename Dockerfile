# ---------- Stage 1: Build Dependencies ----------
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime Image ----------
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
# Create a non-root user
RUN useradd -m appuser
USER appuser
# Set environment variables and expose the port
ENV PORT=8080
EXPOSE 8080
# Start the web service; adjust main:app if your WSGI app is elsewhere.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--timeout", "600", "-k", "eventlet", "main:app"]
