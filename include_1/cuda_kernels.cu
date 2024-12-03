#include <cuda_runtime.h>
#include "cuda_kernels.cuh"
#include <vector>
using namespace std;

namespace hnsw {

// CUDA kernel declarations
__device__ float euclidean_distance_cuda(const float* a, const float* b, int dim) {
    __shared__ float shared_sum[256];  // Using shared memory for reduction
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Each thread processes multiple elements with stride
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    return (tid == 0) ? sqrt(shared_sum[0]) : 0.0f;
}

// Add the kernel implementation
__global__ void batch_distance_calculation(
    const float* queries,
    const float* dataset,
    float* distances,
    int n_queries,
    int n_points,
    int dim
) {
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

class AsyncDistanceCalculator {
private:
    float *d_vectors = nullptr;  // Combined buffer for all vectors
    float *d_distances = nullptr;
    float *h_distances = nullptr;  // Pinned memory
    cudaStream_t stream;
    int current_capacity = 0;
    int vector_dim = 0;

public:
    AsyncDistanceCalculator() {
        cudaStreamCreate(&stream);
        cudaMallocHost(&h_distances, 1024 * sizeof(float));  // Pre-allocate for batch processing
    }

    ~AsyncDistanceCalculator() {
        if (d_vectors) cudaFree(d_vectors);
        if (d_distances) cudaFree(d_distances);
        if (h_distances) cudaFreeHost(h_distances);
        cudaStreamDestroy(stream);
    }

    void ensure_device_memory(int batch_size, int dim) {
        int required_capacity = batch_size * 2 * dim;  // Space for pairs of vectors
        if (required_capacity > current_capacity || dim != vector_dim) {
            if (d_vectors) cudaFree(d_vectors);
            if (d_distances) cudaFree(d_distances);
            
            cudaMalloc(&d_vectors, required_capacity * sizeof(float));
            cudaMalloc(&d_distances, batch_size * sizeof(float));
            
            current_capacity = required_capacity;
            vector_dim = dim;
        }
    }

    float calculate_distance(const vector<float>& p1, const vector<float>& p2) {
        int dim = p1.size();
        ensure_device_memory(1, dim);

        // Copy both vectors in one transfer
        cudaMemcpyAsync(d_vectors, p1.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vectors + dim, p2.data(), dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);

        // Launch kernel with optimal configuration
        int threadsPerBlock = 256;
        batch_distance_calculation<<<1, threadsPerBlock, 0, stream>>>(
            d_vectors, d_vectors + dim, d_distances, 1, 1, dim);

        // Async copy result
        cudaMemcpyAsync(h_distances, d_distances, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        return h_distances[0];
    }
};

// Static instance for reuse
static AsyncDistanceCalculator calculator;

float cuda_euclidean_distance(const vector<float>& p1, const vector<float>& p2) {
    return calculator.calculate_distance(p1, p2);
}

vector<float> batch_cuda_euclidean_distance(
    const vector<vector<float>>& vectors1,
    const vector<vector<float>>& vectors2,
    int batch_size
) {
    if (vectors1.empty() || vectors2.empty() || vectors1.size() != vectors2.size()) {
        return vector<float>();
    }

    int dim = vectors1[0].size();
    vector<float> results(batch_size);
    
    // Flatten input vectors
    vector<float> flat_vectors1, flat_vectors2;
    flat_vectors1.reserve(batch_size * dim);
    flat_vectors2.reserve(batch_size * dim);
    
    for (int i = 0; i < batch_size; i++) {
        flat_vectors1.insert(flat_vectors1.end(), vectors1[i].begin(), vectors1[i].end());
        flat_vectors2.insert(flat_vectors2.end(), vectors2[i].begin(), vectors2[i].end());
    }
    
    // Allocate device memory
    float *d_vec1, *d_vec2, *d_results;
    cudaMalloc(&d_vec1, flat_vectors1.size() * sizeof(float));
    cudaMalloc(&d_vec2, flat_vectors2.size() * sizeof(float));
    cudaMalloc(&d_results, batch_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_vec1, flat_vectors1.data(), flat_vectors1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, flat_vectors2.data(), flat_vectors2.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    batch_distance_calculation<<<blocksPerGrid, threadsPerBlock>>>(
        d_vec1, d_vec2, d_results, batch_size, 1, dim);
    
    // Copy results back
    cudaMemcpy(results.data(), d_results, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_results);
    
    return results;
}

} // namespace hnsw 