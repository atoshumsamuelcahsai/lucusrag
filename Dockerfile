FROM python:3.11-slim


WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Add local bin to PATH for pip-installed scripts
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Switch to non-root user
USER appuser

COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --index-url https://pypi.org/simple --no-cache-dir -r requirements.txt && \
    pip install --index-url https://pypi.org/simple --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY rag/ ./rag/
COPY test/ ./test/
COPY server.py ./
COPY run_server.py ./
COPY templates/ ./templates/
COPY pyproject.toml ./
COPY .pre-commit-config.yaml ./

# Change ownership of all copied files to appuser
USER root
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for FastAPI
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

