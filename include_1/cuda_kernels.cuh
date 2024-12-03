#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace hnsw {

__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim);

__global__ void batch_euclidean_distance(
    const float* vectors,
    const int* pairs,
    float* results,
    int num_pairs,
    int dim
);

float cuda_euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2);

} // namespace hnsw