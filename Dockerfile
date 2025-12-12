# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including gcloud CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && curl -sSL https://sdk.cloud.google.com | bash \
    && rm -rf /var/lib/apt/lists/*

# Add gcloud to PATH
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

# Copy requirements file
COPY requirements-streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy application files
COPY Model-Pipeline/ ./Model-Pipeline/
COPY app.py .
COPY download_models.sh .

# Make download script executable
RUN chmod +x download_models.sh

# Create necessary directories
RUN mkdir -p Model-Pipeline/models Model-Pipeline/results mlruns

# Expose port for Cloud Run
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Download models and run Streamlit
CMD ["sh", "-c", "./download_models.sh && streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true --browser.serverAddress=0.0.0.0 --browser.gatherUsageStats=false"]