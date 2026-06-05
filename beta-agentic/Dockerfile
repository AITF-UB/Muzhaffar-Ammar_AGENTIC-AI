FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for building some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libxcb1 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxdamage1 \
    libxrandr2 \
    libxi6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports for both agentic (8002) and upload (8005)
EXPOSE 8002 8005

# Run with Gunicorn and Uvicorn workers
# Note: we use -w 2 by default to keep RAM usage low for local embeddings
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "600"]
