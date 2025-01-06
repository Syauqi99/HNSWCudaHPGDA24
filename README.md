# High-Performance Data & Graph Analytics - Fall 2024

This repository contains the implementation of Hierarchical Navigable Small World (HNSW) using CUDA for the High-Performance Data and Graph Analytics contest (Between Polimi & Oracle). For the experiment we provide already some google colab notebooks to run the program with the experiments provided in the report, the small and big dataset as well. This will avoid you to build the project and run the program from local.

**Professors**: Ian Di Dio Lavore, Leonardo De Grandis, and Riccardo Strina

# Running Online
## Google Colab

- **Experiment Results**: [View results on Colab](https://colab.research.google.com/drive/1mJX1L5YP1NI6FhohObpJhJZx6jm-Z5Xh#scrollTo=Db17kLYGQw2M)
- **Run with Big Data**: [Big Data Colab](https://drive.google.com/file/u/0/d/1mxvrA9AfZvaQHW6ppRM_0PK9I_6Ux4Hb/edit)

If you prefer to run the program from local, you can follow the instructions below as you can find in the colab notebooks

# Running Local
## Prerequisites

- CUDA 12.0
- CMake 3.22
- GCC 11
- Python 3.10

You should install NVIDIA Runtime for CUDA


```bash
# Add NVIDIA repository and install latest versions
wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-systems-2023.3.3 nsight-compute-2023.2.1 && \
     rm -rf /var/lib/apt/lists/*

# Setting environment variable path
import os
os.environ["PATH"] = "/opt/nvidia/nsight-systems/2023.3.3/bin" + os.pathsep + \
                    "/opt/nvidia/nsight-compute/2023.2.1" + os.pathsep + \
                    "/usr/local/bin" + os.pathsep + \
                    os.getenv("PATH")
```

Also be sure to install Cmake g++

```bash
apt-get update
apt-get install -y cmake g++
# Install the build-essential package, which includes g++
apt-get install -y build-essential
```



## Running the Program

The program accepts several parameters for HNSW, which can be specified when running `main_cuda_test`. The parameters are:

- `$k`: Number of nearest neighbors
- `$m`: Maximum number of connections per element
- `$ef_construction`: Size of the dynamic list for the nearest neighbors during construction
- `$ef`: Size of the dynamic list for the nearest neighbors during search
- `$n`: Number of elements
- `$n_query`: Number of queries
- `$repetitions`: Number of repetitions for the experiment

For example usage you can run see next section

## Building the Project

To build the project, use CMake as follows:

```bash
cd /content/hpgda_contest_MM/
rm -rf build
mkdir build && cd build

# Build the project
cmake ..
make

echo "Running Test:"

# Parameters order (k, m, ef_construction, ef, n, n_query, repetitions)
k=100
m=16
ef_construction=100
ef=100
n=1000
n_query=1
repetitions=30

# By defect runs the small dataset

# GPU Model
./main_cuda_test $k $m $ef_construction $ef $n $n_query $repetitions

# CPU Model
#./main $k $m $ef_construction $ef $n $n_query $repetitions

# Choose any main batch to run any configuration
#./main_batch_test
#./main_allocation_test
#./main_stream_test $k $m $ef_construction $ef $n $n_query $repetitions
#./main_pinned_test $k $m $ef_construction $ef $n $n_query $repetitions
#./main_cuda_test $k $m $ef_construction $ef $n $n_query $repetitions
```

for downloading the big dataset you can run the following command:
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvzf sift.tar.gz
```

## Project Folder Structure

The project is organized into several key components, each serving a specific purpose:

1. **Source Files**:
   - **CUDA Implementations**: 
     - `main_cuda_test.cu`: Main CUDA implementation of the HNSW algorithm.
     - `main_batch_test.cu`: Batch processing version.
     - `main_pinned_test.cu`: Pinned memory version.
     - `main_stream_test.cu`: Stream processing version.
     - `main_allocation_test.cu`: Memory allocation version.
   - **CPU Implementation**:
     - `main.cpp`: CPU version of the HNSW algorithm.
   - **Benchmarking**:
     - `benchmark_cuda.cu`: Benchmarks for all implementations.

2. **Include Directories**:
   - `include`: General header files.
   - `include_cuda`: CUDA-specific headers.
   - `include_batch`: Headers for batch processing.
   - `include_stream`: Headers for stream processing.
   - `include_allocation`: Headers for memory allocation.
   - `include_pinned`: Headers for pinned memory.

3. **Datasets**:
   - `datasets/siftsmall`: Contains small dataset files for testing, such as `siftsmall_base.fvecs`, `siftsmall_query.fvecs`, and `siftsmall_groundtruth.ivecs`.

4. **Results**:
   - `results`: Directory for storing output logs and results from experiments.

5. **Build System**:
   - `CMakeLists.txt`: Configuration file for building the project using CMake, specifying build options and dependencies.

6. **Notebooks**:
   - `tutorials`: Jupyter notebooks that we have used to learn CUDA (For our future selfs)
   - `notebooks`: Jupyter notebooks with the experiments and results provided in the report.

## Credits

The C++ implementations are based on [arailly/hnsw](https://github.com/arailly/hnsw) with additional comments and support for parsing `.ivec`/`.fvec` input files.
