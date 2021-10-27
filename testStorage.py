from google.cloud import storage

def download_blob():
    bucket_name = "retina-neural-net.appspot.com"
    source_blob_name = "retina_disease_classifier.model"
    destination_file_name = "./tmp/retina_disease_classifier.model"

    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

download_blob()