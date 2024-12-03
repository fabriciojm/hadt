from google.cloud import storage
from params import *
import pickle

def load_model():
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.get_bucket(BUCKET_NAME_MODELS)

    blobs = list(bucket.list_blobs(prefix="production/pca_xgboost"))
    # try :
    blob = max(blobs, key=lambda x: x.updated)
    latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, blob.name)
    blob.download_to_filename(latest_model_path_to_save)
    latest_model = pickle.load(latest_model_path_to_save)
    print(latest_model)
    #     print("✅ Latest model downloaded from cloud storage")
    #     return latest_model
    # except:
        # print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        # return None

if __name__ == "__main__":
    load_model()
