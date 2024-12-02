#pragma once

#include <cuda_runtime.h>
#include <vector>

// Forward declaration of utils::Data
namespace utils_cuda {
    template <typename T>
    struct Data;
}

namespace hnsw {

// CUDA kernel declarations
__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim);

__global__ void batch_distance_calculation(const float* queries, 
                                        const float* dataset,
                                        float* distances,
                                        int n_queries,
                                        int n_points,
                                        int dim);

// Host functions
void cuda_batch_distance_calculation(const std::vector<float>& queries,
                                   const std::vector<float>& dataset,
                                   std::vector<float>& distances,
                                   int n_queries,
                                   int n_points,
                                   int dim);

// Drop-in replacement for euclidean_distance
template <typename T = float>
float cuda_euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2);
} // namespace hnsw 