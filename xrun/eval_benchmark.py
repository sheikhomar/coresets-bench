import os, subprocess

from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import re

from pathlib import Path

import click

from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse

from xrun.gen import generate_random_seed
from xrun.data.loader import load_dataset
from xrun.data.run_info import RunInfo

BENCHMARK_FILE_NAME = "benchmark-distortion.txt"


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


datasets = dict()
def load_data(run_info: RunInfo):
    dataset_path = run_info.dataset_path
    if dataset_path not in datasets:
        dataset = load_dataset(dataset_path)
        datasets[dataset_path] = dataset
    return datasets[dataset_path]


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
        run_info = load_run_info(file_path.parent)
        already_evaluated = os.path.exists(file_path.parent / BENCHMARK_FILE_NAME)
        if not already_evaluated and run_info is not None:
            return_paths.append(file_path)
    return return_paths


def eval_benchmark(results_dir: str) -> None:
    output_paths = find_unprocesses_result_files(results_dir)
    total_files = len(output_paths)
    for index, result_path in enumerate(output_paths):
        print(f"Processing file {index+1} of {total_files}: {result_path}")
        experiment_dir = result_path.parent

        run_info = load_run_info(experiment_dir)
        if run_info is None:
            print("Cannot process results file because run file is missing.")
            continue

        if not os.path.exists(run_info.dataset_path):
            print(f"Dataset path: {run_info.dataset_path} cannot be found. Skipping...")
            continue

        print(run_info.dataset_path)
        load_data(run_info)

        # unzipped_result_path = unzip_file(result_path)
        # print(f"Successfully computed distortion. Removing {unzipped_result_path}...")
        # os.remove(unzipped_result_path)


@click.command(help="Evaluate algorithm on benchmark dataset.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
def main(results_dir: str) -> None:
    eval_benchmark(results_dir=results_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
