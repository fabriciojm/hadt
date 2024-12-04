import os

# GCP Project
GCP_PROJECT=os.environ.get("GCP_PROJECT")

# Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME=os.environ.get('BUCKET_NAME')
BUCKET_NAME_MODELS=os.environ.get('BUCKET_NAME_MODELS')
LOCAL_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
