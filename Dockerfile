# OperatorForge reproducible benchmark image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install frozen dependencies first for better layer caching.
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy the package source.
COPY . /app

# Default command: run the deterministic demo end-to-end.
CMD ["python", "examples/run_demo.py"]
