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

from sklearn.metrics import pairwise_distances

from xrun.gen import KNOWN_ALGORITHMS, generate_random_seed

KMEANS_PATH = "kmeans/bin/kmeans.exe"


def unzip_file(input_path: Path) -> Path:
    print(f"Unzipping file {input_path}...")
    output_path = Path(os.path.splitext(input_path)[0])
    if not output_path.exists():
        p = subprocess.Popen(
            args=["gunzip", "-k", str(input_path)],
            start_new_session=True
        )
        p.wait()
    assert(output_path.exists())
    return output_path


def compute_centers(result_file_path: Path) -> Path:
    if not os.path.exists(KMEANS_PATH):
        raise Exception(f"Program '{KMEANS_PATH}' cannot be found. You can build it: make -C kmeans")

    with open(result_file_path, 'r') as f:
        line1 = next(f)  # Skip the first line
        line2 = next(f)  # Read point data

    # When counting the number of dimensions for points skip the
    # first entry as it is the weight of the point.
    d = len(line2.split(" ")) - 1  
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
    proc = subprocess.Popen(
        args=command,
        start_new_session=True
    )
    proc.wait()
    return center_path


def compute_weighted_cost(data_file_path: Path, centers_file_path: Path) -> Path:
    print("Computing weighted cost...")
    data = np.loadtxt(fname=data_file_path, dtype=np.double, delimiter=' ', skiprows=1)
    data_weights = data[:,0]
    data_points = data[:,1:]

    centers = np.loadtxt(fname=centers_file_path, dtype=np.double, delimiter=' ', skiprows=0)
    center_weights = centers[:,0] 
    center_points = centers[:,1:]

    # Distances between all data points and center points
    D = pairwise_distances(data_points, center_points, metric="sqeuclidean")

    # For each point (w, p) in S, find the distance to its closest center
    dist_closest_centers = np.min(D, axis=1)

    # Weigh the distances and sum it all up
    weighted_cost = np.sum(data_weights * dist_closest_centers)

    print(f"Computed weighted cost: {weighted_cost}")
    
    cost_file_path = data_file_path.parent / "weighted_cost.txt"
    with open(cost_file_path, "w") as f:
        f.write(str(weighted_cost))
    return cost_file_path


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
    
    total_files = len(output_paths)
    for index, file_path in enumerate(output_paths):
        print(f"Processing file {index+1} of {total_files}...")
        data_file_path = unzip_file(output_paths[0])
        centers_file_path = compute_centers(data_file_path)
        cost_file_path = compute_weighted_cost(data_file_path, centers_file_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
