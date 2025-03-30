# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies AS ROOT first
# Combine steps and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    libpq-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group AFTER installing system packages
# Using groupadd is good practice for chown
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup -m -d /home/appuser appuser
# Ensure home directory exists and has correct permissions (useradd -m should handle this, but belt-and-suspenders)
# RUN mkdir -p /home/appuser && chown appuser:appgroup /home/appuser

# Copy only requirements first, setting ownership
# This can be done as root, the --chown flag handles ownership
COPY --chown=appuser:appgroup requirements.txt .

# Switch to the non-root user NOW
USER appuser

# Set PATH for user installs (effective for the appuser environment)
# Needs to be set before pip install --user is useful for subsequent RUN/CMD steps
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install Python dependencies AS appuser (--user flag)
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
# Since USER is appuser, files will be owned by appuser:appgroup by default in this step
COPY . .

# Expose the port
EXPOSE 8080

# Ensure the entrypoint script is executable (run as appuser)
# Needs to happen after the second COPY
RUN chmod +x entrypoint.sh

# Start the appropriate service using the entrypoint (runs as appuser)
# Use the full path for the script within the container
CMD ["/app/entrypoint.sh"]
