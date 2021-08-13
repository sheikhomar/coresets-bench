from timeit import default_timer as timer
from typing import List

import click
import numpy as np
import pandas as pd

from scipy import linalg
from scipy.sparse import linalg as sparse_linalg, issparse

from xrun.data.loader import load_dataset


def perform_projection(X, target_dim: int):
    print(f"Computing SVD with target dimensions {target_dim}...")
    start_time = timer()

    # U: Unitary matrix having left singular vectors as columns. 
    # s: The singular values, sorted in descending order.
    # V: Unitary matrix having right singular vectors as rows.
    if issparse(X):
        U, s, V = sparse_linalg.eigen.svds(A=X, solver='arpack')
    else:
        U, s, V = linalg.svd(
            a=X,
            full_matrices=False,
            overwrite_a=True,
            lapack_driver='gesdd'
        )

    # Only take the k singular vectors corresponding to the largest singular values
    # by zeroing out the rest of singular values in `s`.
    s[target_dim:] = 0

    # X_transformed = U*diag(s)*V
    X_transformed = np.dot(U, np.dot(np.diag(s), V))

    end_time = timer()
    print(f" - Completed {end_time - start_time:.2f} secs")

    return X_transformed


def persist_to_disk(data: np.ndarray, output_path: str) -> None:
    print(f"Storing data to {output_path}...")
    start_time = timer()
    df_data = pd.DataFrame(data)
    df_data.to_csv(output_path, index=False, header=False)
    end_time = timer()
    print(f" - Completed in : {end_time - start_time:.2f} secs")


def reduce_dim(input_path: str, target_dims: List[int]) -> None:
    X = load_dataset(input_path)
    for target_dim in target_dims:
        X_transformed = perform_projection(X, target_dim)
        output_path = f"{input_path}-svd-d{target_dim}.txt.gz"
        persist_to_disk(X_transformed, output_path)


def validate_target_dims(ctx, param, value):
    if value is None:
        raise Exception("Invalid target dimension")
    ret_val = []
    for s in value.split(","):
        try:
            ret_val.append(int(s))
        except ValueError:
            raise Exception(f"Dimension is not an integer: {s}")
    return ret_val


@click.command(help="Dimensionality Reduction via SVD.")
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-d",
    "--target-dims",
    required=True,
    callback=validate_target_dims
)
def main(input_path: str, target_dims: List[int]):
    reduce_dim(
        input_path=input_path,
        target_dims=target_dims,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
