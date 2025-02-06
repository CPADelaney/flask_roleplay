# ---------- Stage 1: Build Dependencies ----------
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get remove -y gcc libpq-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ---------- Stage 2: Runtime Image ----------
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Copy the entrypoint script into the container and make it executable (as root)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Now create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

ENV PORT=8080
EXPOSE 8080

CMD ["/app/entrypoint.sh"]
