import dataclasses
import os, requests, subprocess, json, shutil, time
from typing import Dict, List
from timeit import default_timer as timer

from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class RunInfo:
    algorithm: str
    dataset: str
    k: int
    m: int
    iteration: int
    randomSeed: int
    output_dir: str
    command: str
    start_time: str
    end_time: str
    duration_secs: int
    process_id: int

    @classmethod
    def load_json(cls, file_path: Path):
        with open(file_path, "r") as f:
            content = json.load(f)
            obj = cls(
                algorithm=content["algorithm"],
                dataset=content["dataset"],
                k=content["k"],
                m=content["m"],
                iteration=content.get("iteration", -1),
                randomSeed=content["randomSeed"],
                output_dir=content.get("output_dir", ""),
                command=content.get("command", ""),
                start_time=content.get("start_time", ""),
                end_time=content.get("end_time", ""),
                duration_secs=content.get("duration_secs", 0),
                process_id=content.get("process_id", -1),
            )
            return obj

    def save_json(self, file_path: Path):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, sort_keys=False)


class ExperimentRunner:
    _datasets = {
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
    _dir_ready = Path("data/queue/ready")
    _dir_in_progress = Path("data/queue/in-progress")
    _dir_completed = Path("data/queue/completed")
    _dir_discarded = Path("data/queue/discarded")

    def __init__(self) -> None:
        for directory in [self._dir_ready, self._dir_in_progress, self._dir_completed, self._dir_discarded]:
            if not directory.exists():
                print(f"Creating directory {directory}...")
                os.makedirs(directory)

    def run(self) -> None:
        self._download_datasets()
        self._lunch_new_run()

    def _lunch_new_run(self) -> None:
        # Find the next run file containing the experiment to run
        while True:
            run_file_path = self._get_next_run_file()
            if self._should_discard(run_file_path):
                self._move_to_discarded(run_file_path)
            else:
                print(f"Will execute experiment from file {run_file_path}")
                break

        # Prepare for launch
        run_file_path = self._move_to_progress(run_file_path)
        run_details: RunInfo = RunInfo.load_json(run_file_path)
        experiment_dir = self._get_experiment_dir(run_details)
        cmd = self._build_command(run_details, experiment_dir)

        # Actual launch
        print(f"Launching experiment with command:\n '{cmd}'")
        p = subprocess.Popen(
            args=cmd,
            stdout=open(experiment_dir / "stdout.out", "a"),
            stderr=open(experiment_dir / "stderr.out", "a"),
            start_new_session=True
        )

        # Create PID file.
        with open(experiment_dir / "pid.out", "w") as f:
            f.write(str(p.pid))

        # Persist run details to disk.
        run_details.output_dir = str(experiment_dir)
        run_details.command = cmd
        run_details.start_time = datetime.now().isoformat()
        run_details.process_id = p.pid
        run_details.save_json(run_file_path)

    def _build_command(self, run: RunInfo, experiment_dir: Path) -> List[str]:
        data_file_path = self._datasets[run.dataset].local_file_path

        if run.algorithm == "bico":
            algorithm_exe_path = "bico/bin/BICO_Quickstart.exe"
            cmd = [
                "nohup",
                algorithm_exe_path,
                run.dataset, # Dataset
                str(data_file_path), # Input path
                str(run.k), # Number of clusters
                str(run.m), # Coreset size
                str(run.randomSeed), # Random Seed
                str(experiment_dir), # Output dir
            ]
            return cmd
        else:
            raise f"Unknown algorithm: {run.algorithm}"
        pass

    def _get_experiment_dir(self, run: RunInfo) -> Path:
        experiment_no = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = f"data/experiments/{run.dataset}/{run.algorithm}-k{run.k}-m{run.m}/{experiment_no}"
        os.makedirs(experiment_dir)
        return Path(experiment_dir)

    def _move_to_progress(self, run_file_path: Path) -> Path:
        inprogress_path = self._dir_in_progress / run_file_path.name
        shutil.move(run_file_path, inprogress_path)
        return inprogress_path

    def _should_discard(self, run_file_path: Path) -> bool:
        paths_to_check = [
            self._dir_in_progress / run_file_path.name,
            self._dir_completed / run_file_path.name,
        ]
        for p in paths_to_check:
            if p.exists():
                return True
        return False

    def _move_to_discarded(self, run_file_path: Path) -> None:
        discarded_file_name = run_file_path.name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        discarded_path = self._dir_discarded / discarded_file_name
        shutil.move(run_file_path, discarded_path)
        print(f"Experiment {run_file_path.name} is in progress or completed. Moving to {discarded_path}...")

    def _get_next_run_file(self) -> Path:
        file_paths = list(self._dir_ready.glob("*.json"))
        print(f"Directory  {self._dir_ready} contains {len(file_paths)} file(s).")
        return file_paths[0]
    
    def _download_datasets(self) -> None:
        for _, dataset in self._datasets.items():
            self._ensure_dataset_exists(dataset)

    def _ensure_dataset_exists(self, dataset: Dataset):
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
            self._download_file(dataset.download_url, local_file_path)

    def _download_file(self, url: str, file_path: Path):
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


@click.command(help="Run experiments.")
def main():
    ExperimentRunner().run()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
