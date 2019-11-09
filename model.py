import os
from tqdm import tqdm

# Cloud interface
from google.cloud import storage



def main():
    credential_path: str = "/Users/michaeldac/Downloads/mouse-labeler-cff0443f5b5e.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    GCP_PROJECT_NAME: str = 'mouse-labeler'
    GCP_BUCKET_NAME: str =  'skull-images'

    DOWNLOADED_IMAGE_DIRECTORY: str = "/Users/michaeldac/Code/CUNY/698/Downloaded_Skulls/"

    if not os.path.exists(DOWNLOADED_IMAGE_DIRECTORY):
        os.makedirs(DOWNLOADED_IMAGE_DIRECTORY)

    download_blobs(DOWNLOADED_IMAGE_DIRECTORY, GCP_PROJECT_NAME, GCP_BUCKET_NAME)



def download_blobs(folder: str, project_name: str, bucket_name: str):
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    count = 0
    for blob in tqdm(blobs):
        # print(blob.name)
        destination_filename = f'{folder}{blob.name}'
        blob.download_to_filename(destination_filename)
        count += 1
    
    print('\n', f'{count} blobs were downloaded to {folder}') 




if __name__ == '__main__':
    main()