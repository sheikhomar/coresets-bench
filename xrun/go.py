import os, requests, subprocess

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import click
from tqdm import tqdm

@dataclass
class Dataset:
    name: str
    download_url: str
    file_size: int

    @property
    def local_file_name(self) -> str:
        return os.path.basename(self.download_url)

    @property
    def local_file_path(self) -> Path:
        return Path(f"data/input/{self.local_file_name}")


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


def ensure_dataset_exists(dataset: Dataset):
    local_file_path = dataset.local_file_path

    if not local_file_path.parent.exists():
        os.makedirs(str(local_file_path.parent))

    if local_file_path.exists():
        actual_file_size = local_file_path.stat().st_size
        expected_file_size = dataset.file_size
        if actual_file_size < expected_file_size:
            print(f"The size of file {local_file_path.name} is {actual_file_size} but expected {expected_file_size}. Removing file...")
            os.remove(local_file_path)
    
    if not local_file_path.exists():
        download_file(dataset.download_url, local_file_path)


@click.command(help="Import product data.")
def main():
    datasets = {
        "census": Dataset(
                    name="census",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt",
                    file_size=361344227
                ),
        "covertype": Dataset(
                    name="covertype",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                    file_size=11240707
                ),
        "enron": Dataset(
                    name="enron",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz",
                    file_size=12313965
                ),
        "tower": Dataset(
                    name="tower",
                    download_url="http://homepages.uni-paderborn.de/frahling/instances/Tower.txt",
                    file_size=52828754
                )
    }

    algorithms = {
        "bico": "bico/bin/BICO_Quickstart.exe"
    }

    algorithm = "bico"
    dataset_name = "tower"
    k = 10
    m = 200 * k

    algorithm_exe_path = algorithms[algorithm]
    dataset = datasets[dataset_name]
    ensure_dataset_exists(dataset)
    data_file_path = str(dataset.local_file_path)

    experiment_no = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = f"data/experiments/{dataset_name}/{algorithm}-k{k}-m{m}/{experiment_no}"
    os.makedirs(experiment_dir)

    cmd = [
        "nohup",
        algorithm_exe_path,
        dataset_name, # Dataset
        data_file_path, # Input path
        str(k), # Number of clusters
        str(m), # Coreset size
        "-1", # Random Seed
        experiment_dir, # Output dir
    ]

    print(f"Running experiment with {cmd}")
    p = subprocess.Popen(
        args=cmd,
        stdout=open(experiment_dir + "/stdout.out", "a"),
        stderr=open(experiment_dir + "/stderr.out", "a"),
        start_new_session=True
    )
    with open(experiment_dir + "/pid.out", "w") as f:
        f.write(str(p.pid))
        print(f"Process ID: {p.pid}")
    
    print("Done!")

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
