# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Add a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Install OS dependencies (only PostgreSQL client dev files)
# Combine steps and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    libpq-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Add Python user bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Expose the port
EXPOSE 8080

# Ensure the entrypoint script is executable (already copied)
RUN chmod +x entrypoint.sh

# Start the appropriate service using the entrypoint.
# Use the full path for the script within the container
CMD ["/app/entrypoint.sh"]
