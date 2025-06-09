"""
Download and prepare the IPF (Idiopathic Pulmonary Fibrosis) dataset.
This script handles downloading, extracting, and organizing the dataset files.
"""

import os
import gdown
import zipfile
import shutil
from tqdm import tqdm

def download_file_from_google_drive(file_id, dest_path):
    """
    Download a file from Google Drive using gdown.
    
    Args:
        file_id (str): Google Drive file ID
        dest_path (str): Destination path to save the file
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, dest_path, quiet=False)

def extract_zip(zip_path, extract_path):
    """
    Extract a zip file to the specified path.
    
    Args:
        zip_path (str): Path to the zip file
        extract_path (str): Path to extract the contents to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def main():
    """Main function to download and prepare the dataset."""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download and extract training data
    print("Downloading training data...")
    download_file_from_google_drive('1-0C3HFc6-dw-5YQw9z9J9Z9Z9Z9Z9Z9Z9', 'data/train.zip')
    extract_zip('data/train.zip', 'data')
    os.remove('data/train.zip')
    
    # Download and extract validation data
    print("Downloading validation data...")
    download_file_from_google_drive('1-0C3HFc6-dw-5YQw9z9J9Z9Z9Z9Z9Z9Z9', 'data/val.zip')
    extract_zip('data/val.zip', 'data')
    os.remove('data/val.zip')
    
    # Download and extract test data
    print("Downloading test data...")
    download_file_from_google_drive('1-0C3HFc6-dw-5YQw9z9J9Z9Z9Z9Z9Z9Z9', 'data/test.zip')
    extract_zip('data/test.zip', 'data')
    os.remove('data/test.zip')
    
    print("Dataset preparation completed!")

if __name__ == '__main__':
    main() 