# Use a safe, lightweight, updated Python image
FROM python:3.13-slim

# Set environment variables (important for Windows and GCP compatibility)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT=8080

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud BigQuery dependencies early
RUN pip install --no-cache-dir google-cloud-bigquery pandas

# Copy the requirements first (use cache if no changes)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code (including /pages folder)
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Set the startup command to run your Dash app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "multi_page_app:app"]
