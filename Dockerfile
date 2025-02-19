# ---------- Stage 1: Build Dependencies ----------
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

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

# Create a non-root user
RUN useradd -m appuser

# Override any inherited ENTRYPOINT/CMD so Fly.io can use its process definitions
ENTRYPOINT []
CMD []

# **Add the hosts entry for CloudAMQP**
RUN echo "54.193.232.128 duck.lmq.cloudamqp.com" >> /etc/hosts

ENV PORT=8080
EXPOSE 8080
