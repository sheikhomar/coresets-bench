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

The k-means++ tool can be built with Make:

```bash
make -C kmeans
```

The GS project can be built with CMake:

```bash
sudo apt-get update
sudo apt-get install -y ninja-build
cmake -S gs -B gs/build -G "Ninja"
cmake --build gs/build
```

## Debugging

### Segmentation fault

Use AddressSanitizer (ASAN) to debug segfaults. ASAN can help detect memory errors at runtime.

```bash
sudo apt install libgcc-9-dev
g++ -ggdb -std=c++17 -fsanitize=address -std=c++17 -o bin/rp.exe main.cpp
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
