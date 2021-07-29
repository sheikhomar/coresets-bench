from timeit import default_timer as timer
from typing import List

import click
import numpy as np

from sklearn.decomposition import TruncatedSVD

from xrun.data.loader import load_dataset


def perform_projection(X, target_dim: int, input_path: str):
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


def reduce_dim(input_path: str, target_dims: List[int]) -> None:
    X = load_dataset(input_path)
    for target_dim in target_dims:
        perform_projection(X, target_dim, input_path)


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
