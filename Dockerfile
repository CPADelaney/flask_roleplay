########################################
# Stage 1: Build dependencies (optional #
# multi-stage for smaller final image)  #
########################################

# Note: If you want a single‐stage build, skip this entire builder stage,
# and move the combined RUN into the final stage. See Section 5 below.
FROM python:3.10-slim AS builder

##############################
# A. Set environment for pip #
##############################
ENV PIP_DISABLE_PIP_VERSION_CHECK=1   # [1] :contentReference[oaicite:0]{index=0}

WORKDIR /app

##############################################
# B. Install OS dependencies for building    #
#    C extensions (psycopg2, faiss, spacy, etc.) #
##############################################
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \                           # [2] Needed for C/Cython extensions (psycopg2, faiss, etc.) :contentReference[oaicite:1]{index=1}
      pkg-config \                    # [3] Helps locate header files for native libs :contentReference[oaicite:2]{index=2}
      default-libmysqlclient-dev \     # [4] MySQL client headers for mysqlclient :contentReference[oaicite:3]{index=3}
      libpq-dev \                     # [5] PostgreSQL client headers for psycopg2 :contentReference[oaicite:4]{index=4}
      git \                           # [6] If you need to pull from a private repository 
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*       # [7] Clean up apt caches to reduce image size :contentReference[oaicite:6]{index=6}

####################################################
# C. Copy dependency files and install Python deps #
####################################################
# Copy both requirements.txt and constraints.txt
COPY requirements.txt constraints.txt .   # [8] Leverage Docker cache: if neither changes, pip install is reused :contentReference[oaicite:7]{index=7}

# Upgrade pip/setuptools/wheel, then install everything (deps + spaCy model)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt -c constraints.txt \
 && python -m spacy download en_core_web_sm   # [9] spaCy’s small English model baked into the image :contentReference[oaicite:8]{index=8}

#################################################
# Stage 2: Final image (copy only installed deps) #
#################################################
FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1   # [10] :contentReference[oaicite:9]{index=9}

WORKDIR /app

############################
# D. Copy Python packages #
############################
# Copy installed site‐packages from the builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages   # [11] Copies dependencies (including spaCy, PyMC, etc.) :contentReference[oaicite:10]{index=10}
COPY --from=builder /usr/local/bin /usr/local/bin   # [12] Copies any console scripts (e.g., spaCy CLI) :contentReference[oaicite:11]{index=11}

###########################
# E. Copy application code #
###########################
COPY . .   # [13] Copy all application files (after deps installed, for caching) :contentReference[oaicite:12]{index=12}

#################################
# F. Make entrypoint executable #
#################################
RUN chmod +x entrypoint.sh   # [14] :contentReference[oaicite:13]{index=13}

###################
# G. Expose the port #
###################
EXPOSE 8080   # [15] Documents container listening port :contentReference[oaicite:14]{index=14}

###########################
# H. Set the entrypoint   #
###########################
ENTRYPOINT ["/app/entrypoint.sh"]   # [16] Container launches via entrypoint script :contentReference[oaicite:15]{index=15}
