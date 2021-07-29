import gzip
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.sparse.csr import csr_matrix


def load_census_dataset(file_path: str):
    print(f"Loading Census data from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=1,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    return data[:,1:]


def load_tower_dataset(file_path: str):
    print(f"Loading Tower dataset from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=0,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    
    D = 3
    N = int(data.shape[0] / D)
    return data.reshape((N, D))


def load_covertype_dataset(file_path: str):
    print(f"Loading Covertype dataset from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=0,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    return data[:, 0:-1] # Skip the last column which is the classification column


def extract_bow_shape(input_path: str) -> Tuple[int, int]:
    """Extracts the first two lines of the given BoW file.

    The format of the BoW files is 3 header lines, followed by data triples:
    ---
    N    -> the number of documents
    D    -> the number of words in the vocabulary
    NNZ  -> the number of nonzero counts in the bag-of-words
    docID wordID count
    docID wordID count
    ...
    docID wordID count
    docID wordID count
    ---
    """
    shape = []
    with gzip.open(input_path,'rt') as f:
        shape = [int(next(f)) for _ in range(2)]
    return (shape[0], shape[1])


def load_bag_of_words_dataset(input_path: str) -> csr_matrix:
    print(f"Loading BoW dataset from {input_path}")

    data_shape = extract_bow_shape(input_path)
    print(f"Data shape: {data_shape}")

    start_time = timer()
    row_idx, column_idx, values = np.loadtxt(
        fname=input_path,
        dtype=np.uint32,
        delimiter=" ",
        skiprows=3,
        unpack=True
    )
    end_time = timer()
    print(f"Elapsed time: {end_time - start_time:.2f} secs")
    start_time = timer()

    return csr_matrix((values.astype(np.double), (row_idx-1, column_idx-1)), shape=data_shape)


def load_dataset(input_path: str) -> object:
    loader_fn_map : Dict[str, Callable[[str], object]] = {
        "docword": load_bag_of_words_dataset,
        "Tower": load_tower_dataset,
        "USCensus1990": load_census_dataset,
        "covtype2": load_covertype_dataset
    }
    for name_like, loader_fn in loader_fn_map.items():
        if name_like in input_path:
            return loader_fn(input_path)
    raise Exception(f"Cannot parse {input_path} because format is unknown.")
