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

## Datasets

Generate the `nytimes100d` dataset:

```bash
# Download file
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz \
    -O data/input/docword.nytimes.txt.gz
# Perform dimensionality reduction via random projection.
export CPATH=/home/omar/apps/boost_1_76_0
export LIBRARY_PATH=/home/omar/apps/boost_1_76_0/stage/lib
make -C rp && rp/bin/rp.exe \
    reduce-dim \
    data/input/docword.nytimes.txt.gz \
    8192,100 \
    0 \
    1704100552 \
    data/input/docword.nytimes.rp8192-100.txt.gz
```

Generate the `nytimespcalowd` dataset:

```bash
poetry run python -m xrun.data.tsvd -i data/input/docword.nytimes.txt.gz -d 10,20,30,40,50
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
