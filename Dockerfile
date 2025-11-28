# Dockerfile for m2vdb vector database

FROM python:3.12-slim

# Install build dependencies for compilation
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for rust-indexes)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy application code
COPY . .

# Install m2vdb (this builds Rust extensions and registers m2vdb-server command)
# Just use pip directly - simpler and works fine
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /data && chmod 777 /data

# Expose port
EXPOSE 8000

# Environment variables
ENV M2VDB_DATA_DIR=/data \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Run server (could also use: uvicorn m2vdb.server:app --host 0.0.0.0)
CMD ["m2vdb-server", "--host", "0.0.0.0"]
