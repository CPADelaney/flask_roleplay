#############################################
# Stage 1: Builder (compile-time dependencies) #
#############################################

# Use an official Python runtime as a parent image
FROM python:3.10-slim AS builder

# Disable pip's version check for faster installs
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install OS-level dependencies needed for C/Cython extensions:
# - gcc: needed to compile native Python extensions (e.g., psycopg2, faiss) 
# - pkg-config: helps locate header files for native libraries 
# - default-libmysqlclient-dev: provides MySQL client headers 
# - libpq-dev: provides PostgreSQL client headers 
# - git: required if pulling code from repositories at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      pkg-config \
      default-libmysqlclient-dev \
      libpq-dev \
      git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files for layer caching
COPY requirements.txt constraints.txt ./

# Upgrade pip, setuptools, and wheel, install Python dependencies (with constraints),
# then download spaCy’s small English model:
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt -c constraints.txt \
 && python -m spacy download en_core_web_sm

################################
# Stage 2: Final runtime image #
################################

FROM python:3.10-slim

# Again disable pip’s version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy only the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application code
COPY . .

# Ensure that the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose port 8080
EXPOSE 8080

# Use the entrypoint script to start the application
ENTRYPOINT ["/app/entrypoint.sh"]
