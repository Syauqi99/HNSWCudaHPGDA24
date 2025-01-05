# High-Performance Data & Graph Analytics - Fall 2024

Repository for the High-Performance Data and Graph Analytics contest.

In this repository, we implement Hierarchical Navigable Small World by using CUDA. To run this program you can build by using CMAKE. For example:

cmake ..
make

This program can accept the parameters of HNSW on the main_cuda_test. For example as follows:

./main_cuda_test $k $m $ef_construction $ef $n $n_query $repetitions

For all our implementation we run it from Colab using T4 GPU. The link to colab is as follow:

Experiment Results: https://colab.research.google.com/drive/1mJX1L5YP1NI6FhohObpJhJZx6jm-Z5Xh#scrollTo=Db17kLYGQw2M
Run with Big Data: https://drive.google.com/file/u/0/d/1mxvrA9AfZvaQHW6ppRM_0PK9I_6Ux4Hb/edit


## Credits

The c++ implementations are using https://github.com/arailly/hnsw with additional comments and parsing of .ivec/.fvec input files.


