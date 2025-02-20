# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Render will use (e.g., 8080)
EXPOSE 8080

# Ensure the entrypoint script has execute permissions
RUN chmod +x entrypoint.sh

# Use the entrypoint script to start the appropriate service
CMD ["./entrypoint.sh"]
