# Test Bench for Coreset Algorithms

BICO code is downloaded from the [BICO website](https://ls2-www.cs.tu-dortmund.de/grav/en/bico#references).

## Getting Started

Remember to install the prerequisite libraries and tools:

```bash
./install_prerequisites.sh
```

The BICO project can be built by using supplied `Makefile` in the `bico/build` directory:

```bash
make -C bico/build
```

The MT project can be built with Make:

```bash
make -C mt
```

The GS project can be built with CMake:

```bash
cmake -S gs -B gs/build
cmake --build gs/build
```

## Running Experiments

```bash
pyenv install
poetry install
poetry run python -m xrun.go
```

Create conda environment:

```bash
conda env create -f environment.yml 
```
