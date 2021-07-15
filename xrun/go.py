import os
import requests

from pathlib import Path

import click
from tqdm import tqdm


def download_file(url: str, file_path: Path):
    """
    Downloads file from `url` to `file_path`.
    """
    print(f"Downloading {url} to {file_path}...")
    chunk_size = 1024
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        total_size = int(r.headers.get('Content-Length', 10 * chunk_size))
        pbar = tqdm( unit="B", unit_scale=True, total=total_size)
        for chunk in r.iter_content(chunk_size=chunk_size): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)


def ensure_dataset_exists(url: str, expected_file_size: int=1024):
    file_name = os.path.basename(url)
    local_file_path = Path(f"data/raw/{file_name}")

    if not local_file_path.parent.exists():
        os.makedirs(str(local_file_path.parent))

    actual_file_size = local_file_path.stat().st_size
    if local_file_path.exists() and actual_file_size < expected_file_size:
        print(f"The size of file {local_file_path.name} is {actual_file_size} but expected {expected_file_size}. Removing file...")
        os.remove(local_file_path)
    
    if not local_file_path.exists():
        download_file(url, local_file_path)


@click.command(help="Import product data.")
def main():
    ensure_dataset_exists("https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt", 361344227)
    ensure_dataset_exists("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", 11240707)
    ensure_dataset_exists("https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz", 12313965)
    ensure_dataset_exists("http://homepages.uni-paderborn.de/frahling/instances/Tower.txt", 52828754)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
