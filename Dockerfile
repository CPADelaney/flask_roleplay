# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    pkg-config \
    libmariadb-dev \
    libpq-dev

# Copy requirements and install them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that will be used (e.g., 8080).
EXPOSE 8080

# Ensure the entrypoint script is executable.
RUN chmod +x entrypoint.sh

# Start the appropriate service using the entrypoint.
CMD ["./entrypoint.sh"]
