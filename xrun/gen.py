import os, requests, subprocess, json, shutil, time
from typing import List
from timeit import default_timer as timer

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

KNOWN_ALGORITHMS = [
    "basic-clustering",
    "best-clustering",
    "sensitivity-sampling",
    "group-sampling",
    "bico",
]


def validate_algorithms(ctx, param, value):
    if value is None or value == "all":
        return KNOWN_ALGORITHMS
    ret_val = []
    for s in value.split(","):
        if s in ["basic", "basic-clustering"]:
            ret_val.append("basic-clustering")
        elif s in ["best", "best-clustering"]:
            ret_val.append("best-clustering")
        elif s in ["ss", "sensitivity", "sensitivity-sampling"]:
            ret_val.append("sensitivity-sampling")
        elif s in ["gs", "group", "group-sampling"]:
            ret_val.append("group-sampling")
        elif s in ["bico"]:
            ret_val.append("bico")
    return ret_val
        

@click.command(help="Generates experiment files.")
@click.option(
    "-s",
    "--iter-start",
    type=click.INT,
    required=True,
)
@click.option(
    "-e",
    "--iter-end",
    type=click.INT,
    required=True,
)
@click.option(
    "-a",
    "--algorithms",
    type=click.STRING,
    required=False,
    default=None,
    callback=validate_algorithms
)
def main(iter_start: int, iter_end: int, algorithms: List[str]) -> None:
    print(algorithms)
    dataset_k_ranges = {
        "census": [10, 20, 30, 40, 50],
        "covertype": [10, 20, 30, 40, 50],
        "enron": [10, 20, 30, 40, 50],
        "tower": [20, 40, 60, 80, 100],
    }

    ready_dir = Path("data/queue/ready")
    if not ready_dir.exists():
        os.makedirs(ready_dir)
    for dataset, k_values in dataset_k_ranges.items():
        for algo in algorithms:
            for k in k_values:
                for i in range(iter_start, iter_end+1):
                    m = 200 * k
                    exp_details = {
                        "iteration": i,
                        "algorithm": algo,
                        "dataset": dataset,
                        "k": k,
                        "m": m,
                        "randomSeed": -1,
                    }
                    file_path = ready_dir / f"{i:03}-{dataset}-{algo}-k{k}-m{m}.json"
                    if not file_path.exists():
                        print(f"Writing {file_path}...")
                        # with open(file_path, "w") as f:
                        #     json.dump(exp_details, f, indent=4)
                    else:
                        print(f"File already exists {file_path}. Skipping...")

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
