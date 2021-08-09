import os, subprocess

from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import re

from pathlib import Path

import click

from sklearn.metrics import pairwise_distances

from xrun.gen import generate_random_seed
from xrun.data.loader import load_dataset
from xrun.data.run_info import RunInfo

KMEANS_PATH = "kmeans/bin/kmeans.exe"


def unzip_file(input_path: Path) -> Path:
    output_path = Path(os.path.splitext(input_path)[0])
    if not output_path.exists():
        print(f"Unzipping file {input_path}...")
        p = subprocess.Popen(
            args=["gunzip", "-k", str(input_path)],
            start_new_session=True
        )
        p.wait()
    assert(output_path.exists())
    return output_path


def compute_centers(result_file_path: Path) -> Path:
    center_path = result_file_path.parent / "centers.txt"
    
    if center_path.exists() and center_path.stat().st_size > 0:
        return center_path

    if not os.path.exists(KMEANS_PATH):
        raise Exception(f"Program '{KMEANS_PATH}' cannot be found. You can build it: make -C kmeans")

    start_time = timer()

    with open(result_file_path, 'r') as f:
        line1 = next(f)  # Skip the first line
        line2 = next(f)  # Read the first point data to figure out dimensions

    # When counting the number of dimensions for points skip the
    # first entry as it is the weight of the point.
    d = len(line2.split(" ")) - 1  
    k = int(re.findall(r'-k(\d+)-', str(result_file_path))[0])
    random_seed = generate_random_seed()
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
    end_time = timer()
    print(f"k-means++ centers computed in {end_time - start_time:.2f} secs")
    return center_path


datasets = dict()

def load_original_data(run_info: RunInfo):
    dataset_name = run_info.dataset

    if "hardinstance" in dataset_name:
        dataset_name = f"{dataset_name}-k{run_info.k}"
    elif run_info.is_low_dimensional_dataset:
        dataset_name = run_info.original_dataset_name

    if dataset_name not in datasets:
        if run_info.is_low_dimensional_dataset:
            dataset_paths = {
                "census": "data/input/USCensus1990.data.txt",
                "covertype": "data/input/covtype.data.gz",
                "enron": "data/input/docword.enron.txt.gz",
            }
            dataset_path = dataset_paths[dataset_name]
        else:
            dataset_path = run_info.dataset_path
        datasets[dataset_name] = load_dataset(dataset_path)

    return datasets[dataset_name]


def compute_real_cost(experiment_dir: Path, centers_file_path: Path, data_points: np.ndarray) -> Path:
    cost_file_path = experiment_dir / "real_cost.txt"
    if cost_file_path.exists():
        return cost_file_path

    print("Computing real cost... ", end="")

    centers = np.loadtxt(fname=centers_file_path, dtype=np.double, delimiter=' ', skiprows=0)
    center_weights = centers[:,0] 
    center_points = centers[:,1:]

    # Distances between all data points and center points
    D = pairwise_distances(data_points, center_points, metric="sqeuclidean")

    # For each point (w, p) in S, find the distance to its closest center
    dist_closest_centers = np.min(D, axis=1)

    # Weigh the distances and sum it all up
    cost = np.sum(dist_closest_centers)

    print(f"Computed real cost: {cost}")
    
    with open(cost_file_path, "w") as f:
        f.write(str(cost))
    return cost_file_path


def compute_coreset_costs(coreset_file_path: Path, centers_file_path: Path, added_cost: float) -> Path:
    cost_file_path = coreset_file_path.parent / "coreset_cost.txt"
    if cost_file_path.exists():
        return cost_file_path

    print("Computing coreset cost... ", end='')
    coreset = np.loadtxt(fname=coreset_file_path, dtype=np.double, delimiter=' ', skiprows=1)
    coreset_weights = coreset[:,0]
    coreset_points = coreset[:,1:]

    centers = np.loadtxt(fname=centers_file_path, dtype=np.double, delimiter=' ', skiprows=0)
    center_weights = centers[:,0] 
    center_points = centers[:,1:]

    # Distances between all corset points and center points
    D = pairwise_distances(coreset_points, center_points, metric="sqeuclidean")

    # For each point (w, p) in S, find the distance to its closest center
    dist_closest_centers = np.min(D, axis=1)

    # Weigh the distances and sum it all up
    weighted_cost = np.sum(coreset_weights * dist_closest_centers)

    print(f"Computed coreset cost: {weighted_cost}")

    if added_cost > 0:
        weighted_cost += added_cost
        print(f" Adding additional cost: {added_cost}")
    
    with open(cost_file_path, "w") as f:
        f.write(str(weighted_cost))
    return cost_file_path


def load_run_info(experiment_dir: Path) -> Optional[RunInfo]:
    run_file_paths = list(experiment_dir.glob("*.json"))
    if len(run_file_paths) != 1:
        print(f"Expected a single run file in {experiment_dir} but found {len(run_file_paths)} files.")
        return None
    return RunInfo.load_json(run_file_paths[0])


def find_unprocesses_result_files(results_dir: str) -> List[Path]:
    search_dir = Path(results_dir)
    output_paths = list(search_dir.glob('**/results.txt.gz'))
    return_paths = []
    for file_path in output_paths:
        costs_computed = np.all([
            os.path.exists(file_path.parent / cfn)
            for cfn in ["real_cost.txt", "coreset_cost.txt"]
        ])
        if not costs_computed:
            return_paths.append(file_path)
    return return_paths


def get_added_cost_for_low_dimensional_dataset(run_info: RunInfo) -> float:
    if run_info.is_low_dimensional_dataset:
        sqrfrob_file_path = f"{run_info.dataset_path}-sqrfrob.txt"
        if not os.path.exists(sqrfrob_file_path):
            raise Exception(f"File not found: {sqrfrob_file_path} Run `python -m xrun.data.calc_frob_diff -d data/input` to generate extra costs.")
        with open(sqrfrob_file_path, "r") as fp:
            return float(fp.read())
    return 0.0


@click.command(help="Compute costs for coresets.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
def main(results_dir: str) -> None:
    output_paths = find_unprocesses_result_files(results_dir)
    total_files = len(output_paths)
    for index, result_path in enumerate(output_paths):
        print(f"Processing file {index+1} of {total_files}: {result_path}")
        experiment_dir = result_path.parent

        run_info = load_run_info(experiment_dir)
        if run_info is None:
            print("Cannot process results file because run file is missing.")
            continue

        try:
            original_data_points = load_original_data(run_info)
        except FileNotFoundError as ex:
            print(f"Cannot load original dataset! Error: {ex}")
            continue

        added_cost = get_added_cost_for_low_dimensional_dataset(run_info)

        unzipped_result_path = unzip_file(result_path)
        centers_file_path = compute_centers(unzipped_result_path)
        compute_coreset_costs(unzipped_result_path, centers_file_path, added_cost)
        print(f"Successfully computed coreset cost. Removing {unzipped_result_path}...")
        os.remove(unzipped_result_path)
        compute_real_cost(experiment_dir, centers_file_path, original_data_points)
        print(f"Done processing file {index+1} of {total_files}.")
        


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
