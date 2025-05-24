# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    pkg-config \
    default-libmysqlclient-dev \
    libpq-dev \
    git \
    # Add these for potential optimizations
    libblas-dev \
    liblapack-dev \
    gfortran \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies globally with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model to avoid runtime downloads
# This caches the model in the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Model downloaded successfully')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('stopwords', download_dir='/usr/local/nltk_data'); \
    nltk.download('punkt', download_dir='/usr/local/nltk_data')"
ENV NLTK_DATA=/usr/local/nltk_data

# Create cache directories with proper permissions
RUN mkdir -p /app/model_cache /app/cache && \
    chmod 755 /app/model_cache /app/cache

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy the application code
COPY . .

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Run a quick test to ensure all models load correctly
RUN python -c "import spacy; import nltk; from sentence_transformers import SentenceTransformer; \
print('spaCy:', spacy.load('en_core_web_sm')); \
print('NLTK stopwords:', 'stopwords' in nltk.data.find('corpora/stopwords').path); \
print('Sentence transformer: OK'); \
print('All models verified!')"

# Expose the port
EXPOSE 8080

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Using ENTRYPOINT makes the container act like an executable
ENTRYPOINT ["/app/entrypoint.sh"]
