
# Import the Google Auth Library
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import time
import warnings

import os 
import pandas as pd

import io

from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def authenticate(servicePath: str) -> service_account.Credentials:
    """
    Fetch the credentials of the service account from the json file

    Args:
        servicePath (str): The path to the service account json file

    Returns:
        service_account.Credentials: The credentials of the service account
    """
    credentials = service_account.Credentials.from_service_account_file(
        servicePath, scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return credentials


# Credentials
SERVICE_ACCOUNT_CREDENTIAL_PATH = os.getenv("SERVICE_ACCOUNT")  # File that contains credentials of Habibi Service Account
PARENTS_FOLDER_ID = "1-AHq0gILYYjUY0ThwuHIdOzFRLW0OjK0"  # Folder ID of GGSP Drive folder
SERVICE = build("drive", "v3", 
                credentials=authenticate(SERVICE_ACCOUNT_CREDENTIAL_PATH), 
                cache_discovery=False)  # Service account object

# Paths
SOURCE_PATH = os.getenv("SOURCE_PATH")  # Path where the files are stored
DESTINATION_PATH = os.getenv("DESTINATION_PATH")  # Path where the files will be downloaded



def get_file_id(filename: str) -> str:
    """
    Fetch the file id of a file in the GGSP Drive and return None if the file is not found

    Args:
        filename (str): The name of the file

    Returns:
        str: The file id of the file
    """
    try:
        query = (
            f"name='{filename}' and '{PARENTS_FOLDER_ID}' in parents and trashed=false"
        )
        results = (
            SERVICE.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        items = results.get("files", [])
        file = items[0]["id"]
        print(f"[{filename}] FileId found in GGSP Drive")
        return file
    except Exception as E:
        warnings.warn(f"[{filename}] File not found in GGSP Drive")
        return None


def get_all_file_id() -> dict:
    """
    Fetch the file id for all the files in the GGSP Drive

    Returns:
        dict: The dictionary of the file id and the file name
    """
    print("Fetching all the files in the GGSP Drive")
    results = (
        SERVICE.files()
        .list(
            q=f"'{PARENTS_FOLDER_ID}' in parents and trashed=false",
            spaces="drive",
            fields="files(id, name)",
        )
        .execute()
    )
    items = results.get("files", [])
    return {item["id"]: item["name"] for item in items}


def upload_file(filename: str) -> str:
    """
    Upload the file from local folder to the GGSP Drive folder

    Args:
        filename (str): The filename of the file you want to upload
    """
    # Extract the filename and the file id if it exists
    filename = os.path.basename(filename)
    filepath = os.path.join(SOURCE_PATH, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(DESTINATION_PATH, filename)
    file_id = get_file_id(filename)

    # Convert the file to a media object (that can be forwarded)
    print(f"[{filename}] Uploading the file to GGSP Drive")
    media = MediaFileUpload(filepath, mimetype="application/octet-stream", resumable=True)

    try:
        if file_id is not None:
            # Update the file
            SERVICE.files().update(fileId=file_id, media_body=media).execute()
            print(f"[{filename}] File updated successfully.")
        else:
            # Create the file
            file_metadata = {"name": filename, "parents": [PARENTS_FOLDER_ID]}
            SERVICE.files().create(body=file_metadata, media_body=media).execute()
            print(f"[{filename}] File uploaded successfully.")
    except Exception as E:      
        warnings.warn(f"[{filename}] An error occurred while uploading the file: {E}")

    time.sleep(10)
    return filename


def download_file(filename: str, save=True) -> str:
    """
    Download a file from the GGSP Drive to a local folder

    Args:
        filename (str): The name of the file you want to download from the Drive to the destination local folder
    """
    # Define the destination path
    filename = os.path.basename(filename)
    destination_file_path = os.path.join(DESTINATION_PATH, filename)

    # Fetch the file id
    file_id = get_file_id(filename)
    if file_id is None:
        warnings.warn(f"[{filename}] File not found in GGSP Drive")
        return
    else:
        print(f"[{filename}] File found in GGSP Drive")

    # Request the file
    request = SERVICE.files().get_media(fileId=file_id)
    if save:
        file_buffer = io.FileIO(destination_file_path, "wb")
    else:
        file_buffer = io.BytesIO()

    # Download the file with a downloader
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        try:
            status, done = downloader.next_chunk()
            if status:
                print(f"[{filename}] Download {int(status.progress() * 100)}%.")
        except Exception as e:
            warnings.warn(f"[{filename}] An error occurred during the download of the file: {e}")
            break

    time.sleep(3)

    if save:
        return filename
    if not save:
        try:
            df = pd.read_parquet(file_buffer)
            print(f"[{filename}] File read into DataFrame successfully.")
            return df
        except Exception as e:
            warnings.warn(f"[{filename}] An error occurred while reading the file into DataFrame: {e}")
            return None


def upload_all_files() -> None:
    """
    Upload all the files from the local folder to the GGSP Drive folder
    """
    # List all the files in the local folder recursively
    filesToUpload = []
    for root, _, files in os.walk(SOURCE_PATH):
        parquetFiles = [file for file in files if file.endswith(".parquet")]
        for parquetFile in parquetFiles:
            filesToUpload.append(os.path.join(root, parquetFile))

    # Parallelize the upload of the files with ProcessPoolExecutor
    futures = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        for file in filesToUpload:
            futures.append(executor.submit(upload_file, file))
    
    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Task completed for file: {result}")
        except Exception as exc:
            warnings.warn(f'Task generated an exception: {exc}')

    print(f"All tasks have been completed")


def download_all_files() -> None:
    """
    Download all the files from the GGSP Drive folder
    """
    # Fetch all the files in the GGSP Drive
    filesToDownload = get_all_file_id()

    # Parallelize the download of the files with ProcessPoolExecutor
    futures = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        for file in filesToDownload.values():
            futures.append(executor.submit(download_file, file))

    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Task completed for file: {result}")
        except Exception as exc:
            warnings.warn(f'Task generated an exception: {exc}')
        
    print(f"All tasks have been completed")