"""
Cloud Run Configuration for EchoAI
Handles model paths and GCS integration
"""
import os
from pathlib import Path

# Check if running in Cloud Run
IS_CLOUD_RUN = os.getenv('K_SERVICE') is not None

# Base paths
if IS_CLOUD_RUN:
    BASE_DIR = Path('/app')
    # In Cloud Run, download models from GCS on startup
    GCS_BUCKET = os.getenv('GCS_BUCKET', 'us-central1-echo-ai-9c58f7b7-bucket')
    MODEL_GCS_PATH = f'gs://{GCS_BUCKET}/models/'
else:
    BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR / 'Model-Pipeline' / 'models'
RESULTS_DIR = BASE_DIR / 'Model-Pipeline' / 'results'

# Model paths
BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
VECTORIZER_PATH = MODEL_DIR / 'tfidf_vectorizer.pkl'

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def download_models_from_gcs():
    """Download models from GCS if in Cloud Run"""
    if not IS_CLOUD_RUN:
        return
    
    from google.cloud import storage
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        # Download best_model.pkl
        if not BEST_MODEL_PATH.exists():
            logger.info("Downloading best_model.pkl from GCS...")
            blob = bucket.blob('models/best_model.pkl')
            blob.download_to_filename(BEST_MODEL_PATH)
            logger.info("✓ Downloaded best_model.pkl")
        
        # Download vectorizer
        if not VECTORIZER_PATH.exists():
            logger.info("Downloading tfidf_vectorizer.pkl from GCS...")
            blob = bucket.blob('models/tfidf_vectorizer.pkl')
            blob.download_to_filename(VECTORIZER_PATH)
            logger.info("✓ Downloaded tfidf_vectorizer.pkl")
            
    except Exception as e:
        logger.error(f"Failed to download models from GCS: {e}")
        raise

# Auto-download models on import if in Cloud Run
if IS_CLOUD_RUN:
    download_models_from_gcs()