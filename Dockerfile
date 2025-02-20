FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libpq5 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Set environment variable and expose the port
ENV PORT=8080
EXPOSE 8080

# Start the web service
CMD sh -c "gunicorn --bind 0.0.0.0:$PORT --timeout 600 -k eventlet main:app"
