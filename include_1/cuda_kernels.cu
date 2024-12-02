#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include <vector>
using namespace std;

namespace hnsw {

// CUDA kernel declarations
__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__ void batch_distance_calculation(const float* queries, 
                                        const float* dataset,
                                        float* distances,
                                        int n_queries,
                                        int n_points,
                                        int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_queries * n_points) {
        int query_idx = idx / n_points;
        int point_idx = idx % n_points;
        
        distances[idx] = euclidean_distance_cuda(
            &queries[query_idx * dim],
            &dataset[point_idx * dim],
            dim
        );
    }
}

void cuda_batch_distance_calculation(const std::vector<float>& queries,
                                   const std::vector<float>& dataset,
                                   std::vector<float>& distances,
                                   int n_queries,
                                   int n_points,
                                   int dim) {
    // Allocate device memory
    float *d_queries, *d_dataset, *d_distances;
    cudaMalloc(&d_queries, queries.size() * sizeof(float));
    cudaMalloc(&d_dataset, dataset.size() * sizeof(float));
    cudaMalloc(&d_distances, distances.size() * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset, dataset.data(), dataset.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_queries * n_points + threadsPerBlock - 1) / threadsPerBlock;
    batch_distance_calculation<<<blocksPerGrid, threadsPerBlock>>>(
        d_queries, d_dataset, d_distances, n_queries, n_points, dim);

    // Copy results back to host
    cudaMemcpy(distances.data(), d_distances, distances.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_queries);
    cudaFree(d_dataset);
    cudaFree(d_distances);
}

// Implementation of the template function
template <typename T>
float cuda_euclidean_distance(const vector<float>& p1, const vector<float>& p2) {
    int dim = p1.size();
    vector<float> result(1);

    // Allocate device memory
    float *d_vec1, *d_vec2, *d_result;
    cudaMalloc(&d_vec1, dim * sizeof(float));
    cudaMalloc(&d_vec2, dim * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_vec1, p1.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, p2.data(), dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;
    batch_distance_calculation<<<blocksPerGrid, threadsPerBlock>>>(
        d_vec1, d_vec2, d_result, 1, 1, dim);

    // Copy result back to host
    cudaMemcpy(result.data(), d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);

    return result[0];
}

// Explicit template instantiation
template float cuda_euclidean_distance<float>(const vector<float>&, const vector<float>&);

} // namespace hnsw 