# High-Performance Data & Graph Analytics - Fall 2024

This repository is dedicated to the High-Performance Data and Graph Analytics contest for Fall 2024.

In this repository, we implement the Hierarchical Navigable Small World (HNSW) algorithm using CUDA for high-performance execution. The code is optimized for GPU processing and can be built using CMake.

## Prerequisites

To build and run the program, ensure you have CMake installed. You can build the project with the following commands:

```bash
cmake ..
make
```

## Usage

The program allows you to run the HNSW algorithm with customizable parameters. The primary executable is `main_cuda_test`, which accepts the following parameters:

```bash
./main_cuda_test $k $m $ef_construction $ef $n $n_query $repetitions
```

Where:
- `$k` represents the number of neighbors.
- `$m` defines the maximum number of connections per node.
- `$ef_construction` and `$ef` control the efficiency of the algorithm.
- `$n` specifies the number of data points.
- `$n_query` is the number of queries to perform.
- `$repetitions` determines the number of times to repeat the experiment for consistency.

## Experiment Environment

Our experiments are conducted on Google Colab using a T4 GPU to ensure high performance and reproducibility. You can view the experiment results using the following link:

- [Experiment Results](https://colab.research.google.com/drive/1mJX1L5YP1NI6FhohObpJhJZx6jm-Z5Xh#scrollTo=Db17kLYGQw2M)

## Big Data Run

For large-scale data experiments, we have also made the dataset available here:

- [Run with Big Data](https://drive.google.com/file/u/0/d/1mxvrA9AfZvaQHW6ppRM_0PK9I_6Ux4Hb/edit)

## Credits

The C++ implementations are based on the original [HNSW repository by arailly](https://github.com/arailly/hnsw), with additional comments and parsing functionality for `.ivec` and `.fvec` input files.
