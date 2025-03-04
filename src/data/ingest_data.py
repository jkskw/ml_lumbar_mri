import os
import zipfile
from pathlib import Path
import kaggle
import yaml


def load_config(config_path: str = "config.yml"):
    """
    Load configuration from a YAML file.

    Args:
      config_path (str): Path to the configuration YAML file.

    Returns:
      dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def authenticate_kaggle():
    """
    Authenticate using Kaggle API. The Kaggle library automatically
    looks for credentials in the file ~/.kaggle/kaggle.json.
    Make sure you have placed your API key there with proper permissions.

    Alternatively, you can set environment variables (KAGGLE_USERNAME and KAGGLE_KEY)
    or use a .env file (which should be excluded from Git).
    """
    kaggle.api.authenticate()


def download_and_extract(competition: str, download_dir: str = "data", extract_dir: str = "data/raw"):
    """
    Downloads the competition files and extracts them into the specified raw data directory.

    Args:
      competition (str): The Kaggle competition name.
      download_dir (str): The directory to download the zip file into.
      extract_dir (str): The directory to extract the contents into.
    """

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"[INFO] Downloading competition files for '{competition}'...")
    kaggle.api.competition_download_files(competition, path=download_dir)

    zip_path = Path(download_dir) / f"{competition}.zip"
    if not zip_path.exists():
        print(f"[ERROR] Downloaded file {zip_path} not found!")
        return

    print(f"[INFO] Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_dir)

    os.remove(zip_path)
    print("[INFO] Extraction complete and zip file removed.")


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Extract relevant settings
    COMPETITION = config["kaggle"]["competition"]
    DOWNLOAD_PATH = config["kaggle"]["download_path"]
    EXTRACT_PATH = config["data"]["raw_path"]

    # Run pipeline
    authenticate_kaggle()
    download_and_extract(competition=COMPETITION,
                         download_dir=DOWNLOAD_PATH, extract_dir=EXTRACT_PATH)
