# Stage 1: Build
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.10-slim

# Install runtime dependencies (libpq5 for PostgreSQL)
RUN apt-get update && apt-get install -y libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Set default port for Railway
ENV PORT=8080
EXPOSE $PORT

# Run as non-root user
RUN useradd -m appuser
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "main:app"]
