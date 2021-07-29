from timeit import default_timer as timer

import click
import numpy as np

from sklearn.decomposition import TruncatedSVD

from xrun.data.loader import load_dataset


def reduce_dim(input_path: str, target_dim: float) -> None:
    X = load_dataset(input_path)

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
