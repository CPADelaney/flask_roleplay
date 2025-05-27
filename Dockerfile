# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    default-libmysqlclient-dev \
    libpq-dev \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies globally (don't use --user flag with root)
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose the port
EXPOSE 8080

# Choose either ENTRYPOINT or CMD, not both
# Using ENTRYPOINT makes the container act like an executable
ENTRYPOINT ["/app/entrypoint.sh"]
