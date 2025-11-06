FROM python:3.11-slim


WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirments.txt requirments-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --index-url https://pypi.org/simple --no-cache-dir -r requirments.txt && \
    pip install --index-url https://pypi.org/simple --no-cache-dir -r requirments-dev.txt

# Copy application code
COPY rag/ ./rag/
COPY test/ ./test/
COPY server.py ./
COPY run_server.py ./
COPY templates/ ./templates/
COPY pyproject.toml ./
COPY .pre-commit-config.yaml ./

# Expose port for FastAPI
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

