# !/bin/bash
set -e

echo "Downloading fresh models from GCS..."

# Always download (remove the if check)
gsutil -m cp gs://trans-scheme-480511-e3-models/models/*.pkl /app/Model-Pipeline/models/
echo "✓ Models downloaded successfully!"