#!/bin/bash
set -e

echo "Checking for models..."

# Download models from GCS if they don't exist
if [ ! -f "/app/Model-Pipeline/models/best_model.pkl" ]; then
    echo "Downloading models from GCS..."
    gsutil -m cp gs://trans-scheme-480511-e3-models/models/*.pkl /app/Model-Pipeline/models/
    echo "✓ Models downloaded successfully!"
else
    echo "✓ Models already exist"
fi