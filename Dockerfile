# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libpq5 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Set environment variables and expose the port (Render provides $PORT)
ENV PORT=8080
EXPOSE 8080

# Start the web service using Gunicorn with eventlet
CMD sh -c "gunicorn --bind 0.0.0.0:$PORT --timeout 600 -k eventlet main:app"
