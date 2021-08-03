import os
from timeit import default_timer as timer

import click
import numpy as np


def generate_benchmark(k: int, alpha: int):
    """Generate benchmark dataset.

    Parameters
    ----------
    k: int
        The size of the initial block which is a square k-by-k matrix.
    alpha: int
        The number of the column blocks.

    Returns
    -------
    np.array
        Returns a matrix of size (k^alpha x alpha*k)
    """
    # Construct top right corner k-by-k block
    diag_entries = np.identity(k, dtype=np.double) * ((k-1)/k)
    off_diag_entries = (~np.eye(k,dtype=bool)) * (-1/k)
    block = diag_entries + off_diag_entries
    
    # Create NxD matrix
    n = k ** alpha
    d = alpha * k
    data = np.zeros((n, d))
    
    n_row_blocks = n // k
    
    # Construct first layer which corresponds to
    # repeating block entries downwards some number of times
    first_layer = np.tile(block, (n_row_blocks, 1))
    
    # Fill the last k columns of the matrix
    data[:,-k:] = first_layer
    
    # Fill the rest by using the copy-stack operation
    for j in range(d-k-1, -1, -1):
        for i in range(0, n, 1):
            copy_i = i // k
            copy_j = j + k
            data[i,j] = data[copy_i, copy_j]
    return data


def gen_benchmark(block_size: int, alpha: int, output_dir: str):
    print("Generating benchmark dataset...")
    start_time = timer()
    dataset = generate_benchmark(
        k=block_size,
        alpha=alpha,
    )
    end_time = timer()
    print(f"Dataset of shape {dataset.shape} generated in {end_time - start_time:.2f} secs")

    print("Storing data on disk...")
    start_time = timer()
    np.savetxt(
        fname=os.path.join(output_dir, f"benchmark-k{block_size}-alpha{alpha}.txt.gz"),
        X=dataset,
        delimiter=",",
    )
    end_time = timer()
    print(f"Elapsed time: {end_time - start_time:.2f} secs")


@click.command(help="Generates benchmark dataset.")
@click.option(
    "-k",
    "--block-size",
    type=click.INT,
    required=True,
)
@click.option(
    "-a",
    "--alpha",
    type=click.INT,
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(block_size: int, alpha: int, output_dir: str):
    gen_benchmark(
        block_size=block_size,
        alpha=alpha,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
