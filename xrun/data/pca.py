from timeit import default_timer as timer

import click
import numpy as np

from sklearn.decomposition import TruncatedSVD, PCA

from xrun.data.loader import load_dataset


def reduce_dim(input_path: str, target_dim: float, algorithm: str) -> None:
    X = load_dataset(input_path)

    if algorithm == "lapack":
        algorithm = "full"
    else:
        algorithm = "arpack"

    pca = PCA(
        n_components=int(target_dim),
        svd_solver=algorithm,
    )

    print(f"Computing PCA with target dimensions {target_dim} using solver {algorithm}...")
    start_time = timer()
    X_reduced = pca.fit_transform(X)
    end_time = timer()
    print(f"Elapsed time: {end_time - start_time:.2f} secs")
    print(f"Explained variance ratios: {np.sum(pca.explained_variance_ratio_):0.4}")

    print(f"Saving transformed data to disk...")
    np.savetxt(
        fname=f"{input_path}-pca-d{target_dim}.txt.gz",
        X=X_reduced,
        delimiter=",",
    )
    np.savez_compressed(
        file=f"{input_path}-pca-d{target_dim}.npz",
        X=X_reduced,
    )
    print(f"Saving components to disk...")
    np.savez_compressed(
        file=f"{input_path}-pca-d{target_dim}-output.npz",
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        mean=pca.mean_,
        n_components=pca.n_components_,
        n_features=pca.n_features_,
        noise_variance=pca.noise_variance_,
    )


@click.command(help="Dimensionality Reduction via PCA.")
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
@click.option(
    "-a",
    "--algorithm",
    type=click.STRING,
    required=True,
    help="Use 'lapack' or 'arpack'"
)
def main(input_path: str, target_dim: int, algorithm: str):
    reduce_dim(
        input_path=input_path,
        target_dim=target_dim,
        algorithm=algorithm
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
