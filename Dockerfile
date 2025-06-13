# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

# Install OS dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    default-libmysqlclient-dev \
    libpq-dev \
    git \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip operations
RUN pip install uv

# Copy constraints file first
COPY constraints.txt .

# Install heavy/stable dependencies first (better layer caching)
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system \
    numpy==2.1.3 \
    pandas==2.2.3 \
    torch==2.7.0 \
    tensorflow==2.19.0

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system \
    -c constraints.txt \
    -r requirements.txt \
    --verbose

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose the port
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
