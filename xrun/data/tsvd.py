import os, requests, subprocess, json, shutil, time
from typing import List, Tuple
from timeit import default_timer as timer

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs


import gzip
from sklearn.decomposition import TruncatedSVD

import click


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


def parse_bag_of_words_dataset(input_path: str) -> csr_matrix:
    print(f"Parsing BoW dataset: {input_path}")

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


def reduce_dim(input_path: str, target_dim: float) -> None:
    if "docword" in input_path:
        X = parse_bag_of_words_dataset(input_path)
        svd = TruncatedSVD(
            n_components=int(target_dim),
            algorithm="arpack",
        )
        
        print(f"Computing SVD with target dimensions {target_dim} using ARPACK...")
        start_time = timer()
        X_reduced = svd.fit_transform(X)
        end_time = timer()
        print(f"Elapsed time: {end_time - start_time:.2f} secs")
        print(f"Explained variance ratios: {np.sum(svd.explained_variance_ratio_):0.4}")

        print(f"Saving transformed data to disk...")
        np.savetxt(
            fname=f"{input_path}-svd-d{target_dim}.txt.gz",
            X=X_reduced,
            delimiter=",",
        )
        np.savez_compressed(
            file=f"{input_path}-svd-d{target_dim}.npz",
            X=X_reduced,
        )
        print(f"Saving components to disk...")
        np.savez_compressed(
            file=f"{input_path}-svd-d{target_dim}-output.npz",
            components=svd.components_,
            explained_variance=svd.explained_variance_,
            explained_variance_ratio=svd.explained_variance_ratio_,
            singular_values=svd.singular_values_,
        )


@click.command(help="Dimensionality Reduction via SVD.")
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-d",
    "--target-dim",
    type=click.INT,
    required=True,
)
def main(input_path: str, target_dim: int):
    reduce_dim(
        input_path=input_path,
        target_dim=target_dim,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
