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

__global__ void compute_distances_kernel(
    const float* query_vector,
    const float* candidate_vectors,
    float* distances,
    const int vector_dim,
    const int n_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    float dist = 0.0f;
    const float* candidate = candidate_vectors + idx * vector_dim;
    
    for (int i = 0; i < vector_dim; i++) {
        float diff = query_vector[i] - candidate[i];
        dist += diff * diff;
    }
    
    distances[idx] = sqrt(dist);
}

void batch_compute_distances(
    SearchLayerCUDAContext& ctx,
    const std::vector<float>& query_vector,
    const std::vector<const std::vector<float>*>& candidate_vectors,
    std::vector<float>& distances,
    int batch_size
) {
    // Copy query vector to GPU (if not already done)
    thrust::copy(query_vector.begin(), query_vector.end(), ctx.d_query_vector.begin());

    // Prepare batch data for GPU
    for (size_t i = 0; i < candidate_vectors.size(); i++) {
        thrust::copy(candidate_vectors[i]->begin(),
                    candidate_vectors[i]->end(),
                    ctx.d_candidate_vectors.begin() + i * query_vector.size());
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (candidate_vectors.size() + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_distances_kernel<<<numBlocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(ctx.d_query_vector.data()),
        thrust::raw_pointer_cast(ctx.d_candidate_vectors.data()),
        thrust::raw_pointer_cast(ctx.d_distances.data()),
        query_vector.size(),
        candidate_vectors.size()
    );

    // Copy results back
    thrust::copy_n(ctx.d_distances.begin(), 
                  candidate_vectors.size(),
                  distances.begin());
}

} // namespace hnsw 