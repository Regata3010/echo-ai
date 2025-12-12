# # Use official Python runtime as base image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements file
# COPY requirements-streamlit.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements-streamlit.txt

# # Copy application files
# COPY Model-Pipeline/ ./Model-Pipeline/
# COPY app.py .

# # Copy trained models directly into image
# COPY Model-Pipeline/models/best_model.pkl ./Model-Pipeline/models/
# COPY Model-Pipeline/models/tfidf_vectorizer.pkl ./Model-Pipeline/models/

# # Create necessary directories
# RUN mkdir -p Model-Pipeline/results mlruns

# # Expose port for Cloud Run
# EXPOSE 8080

# # Health check
# HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# # Run Streamlit app directly (no download script)
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--browser.serverAddress=0.0.0.0", "--browser.gatherUsageStats=false"]

# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Download spaCy model for ABSA
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY Model-Pipeline/ ./Model-Pipeline/
COPY app.py .

# Copy trained models directly into image
COPY Model-Pipeline/models/best_model.pkl ./Model-Pipeline/models/
COPY Model-Pipeline/models/tfidf_vectorizer.pkl ./Model-Pipeline/models/

# Create necessary directories
RUN mkdir -p Model-Pipeline/results mlruns

# Expose port for Cloud Run
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit app directly (no download script)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--browser.serverAddress=0.0.0.0", "--browser.gatherUsageStats=false"]