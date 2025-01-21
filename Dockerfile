# ---------- Stage 1: Build Dependencies ----------
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install any build/system deps needed for pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build deps (optional optimization)
RUN apt-get remove -y gcc libpq-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*


# ---------- Stage 2: Runtime Image ----------
FROM python:3.10-slim

# We'll likely still need libpq5 for runtime Postgres connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Now copy your project files from the local repo to the image
COPY . .

# Switch to a non-root user (optional best practice)
RUN useradd -m appuser
USER appuser

ENV PORT=8080
EXPOSE 8080 

CMD ["sh", "-c", "echo 'PORT is set to:' $PORT && sleep 60"]

