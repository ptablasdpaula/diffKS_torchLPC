import requests
import tarfile
from pathlib import Path
import logging
import time

from paths import NSYNTH_DIR as nsynth_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

NSYNTH_URLS = {
    'train': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz',
    'valid': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz',
    'test': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz'
}

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}")

    response = requests.get(url, stream=True)
    total_size_bytes = int(response.headers.get('content-length', 0))
    total_size_mb = total_size_bytes / (1024 * 1024)  # Convert to MB
    block_size = 1024 * 1024  # 1 MB

    with open(target_path, 'wb') as f:
        downloaded_bytes = 0
        for data in response.iter_content(block_size):
            downloaded_bytes += len(data)
            f.write(data)
            downloaded_mb = downloaded_bytes / (1024 * 1024)
            done = int(50 * downloaded_bytes / total_size_bytes)
            print(
                f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_mb:.2f}MB/{total_size_mb:.2f}MB ({(downloaded_mb / total_size_mb * 100):.1f}%)",
                end='')
    print()


def download_and_extract_nsynth(data_dir, splits=None):
    if splits is None:
        splits = ['train', 'valid', 'test']

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        # Skip if already downloaded
        target_dir = data_dir / f"nsynth-{split}"
        if target_dir.exists():
            print(f"Split {split} already exists at {target_dir}, skipping download")
            continue

        # Download
        download_path = data_dir / f"nsynth-{split}.jsonwav.tar.gz"
        if not download_path.exists():
            download_file(NSYNTH_URLS[split], download_path)

        # Extract
        print(f"Extracting {download_path}")
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)

        print(f"Extracted {split} data to {target_dir}")

        # Optionally remove tar file after extraction
        download_path.unlink()

    return data_dir


if __name__ == "__main__":
    start_time = time.time()
    data_dir = nsynth_dir

    logging.info(f"Starting NSynth dataset download and processing")

    download_and_extract_nsynth(data_dir, splits=['train', 'valid', 'test'])

    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")