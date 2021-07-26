import os, requests, subprocess, json, shutil, time
from typing import List
from timeit import default_timer as timer

import numpy as np
import re

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import gzip
from tqdm import tqdm

from xrun.gen import KNOWN_ALGORITHMS, generate_random_seed

KMEANS_PATH = "kmeans/bin/kmeans.exe"


def unzip_file(input_path: Path) -> Path:
    output_path = Path(os.path.splitext(input_path)[0])
    if not output_path.exists():
        p = subprocess.Popen(
            args=["gunzip", "-k", str(input_path)],
            start_new_session=True
        )
        p.wait()
    assert(output_path.exists())
    return output_path

def compute_centers(result_file_path: Path) -> None:
    if not os.path.exists(KMEANS_PATH):
        raise Exception(f"Program '{KMEANS_PATH}' cannot be found. You can build it: make -C kmeans")

    with open(result_file_path, 'r') as f:
        line1 = next(f)  # Skip the first line
        line2 = next(f)  # Read point data

        # When counting the number of dimensions for points skip the
        # first entry as it is the weight of the point.
        d = len(line2.split(" ")) - 1  
    
    # data = np.loadtxt(fname=result_file_path, dtype=np.double, delimiter=' ', skiprows=1)
    # weights = data[:,0]
    # data_points = data[:,1:]
    # d = data.shape[1] - 1

    k = int(re.findall(r'-k(\d+)-', str(result_file_path))[0])
    random_seed = generate_random_seed()
    center_path = result_file_path.parent / "centers.txt"

    command = [
        KMEANS_PATH,
        str(result_file_path),
        str(k),
        str(d),
        str(center_path),
        "0",
        str(random_seed),
    ]
    print(command)

    p = subprocess.Popen(
        args=command,
        start_new_session=True
    )
    p.wait()
    print("Done!")

        

@click.command(help="Compute costs for coresets.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
def main(results_dir: str) -> None:
    parent_dir = Path(results_dir)
    output_paths = list(parent_dir.glob('**/results.txt.gz'))
    # print(f"Number of result files: {len(output_paths)}")
    # print(output_paths[0:5])

    file_path = unzip_file(output_paths[0])
    print(f"Unzipped file: {file_path}")
    compute_centers(file_path)
    
    

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
