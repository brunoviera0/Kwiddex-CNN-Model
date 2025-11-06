from google.cloud import storage

#project and bucket info
GCP_PROJECT = "sentiment-analysis-379200"  
BUCKET_NAME = "kwiddex-datasets"            

_client = storage.Client(project=GCP_PROJECT)

def get_bucket():
    #Return a GCS bucket object
    return _client.bucket(BUCKET_NAME)

def list_files(prefix=""):
    #List blob names under a given prefix
    bucket = get_bucket()
    return [b.name for b in bucket.list_blobs(prefix=prefix)]

def download_file(blob_name, dest_path):
    #Download one blob to a local destination.
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    import os
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    blob.download_to_filename(dest_path)
