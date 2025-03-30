# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies AS ROOT first
# Combine steps and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    default-libmysqlclient-dev \
    libpq-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# --- Rest of your Dockerfile remains the same ---

# Create a non-root user and group AFTER installing system packages
# Using groupadd is good practice for chown
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup -m -d /home/appuser appuser

# Copy only requirements first, setting ownership
COPY --chown=appuser:appgroup requirements.txt .

# Switch to the non-root user NOW
USER appuser

# Set PATH for user installs
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install Python dependencies AS appuser (--user flag)
# This should now work for mysqlclient if it's in requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8080

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Start the appropriate service using the entrypoint
CMD ["/app/entrypoint.sh"]
