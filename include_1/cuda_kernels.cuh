#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace hnsw {

__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim);

__global__ void batch_distance_calculation(
    const float* queries, 
    const float* dataset,
    float* distances,
    int n_queries,
    int n_points,
    int dim
);

float cuda_euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2);

} // namespace hnsw