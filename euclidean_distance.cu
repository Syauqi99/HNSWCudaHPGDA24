#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

using namespace std::chrono;

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void restVectorsInto(float *result, float *a, float *b, int N)
{
  // Process multiple elements per thread
  const int elementsPerThread = 4;
  int index = (threadIdx.x + blockIdx.x * blockDim.x) * elementsPerThread;
  
  #pragma unroll
  for(int i = 0; i < elementsPerThread && index + i < N; i++)
  {
    int idx = index + i;
    if (idx < N) {
      float diff = a[idx] - b[idx];
      result[idx] = diff * diff;
    }
  }
}

// Kernel for optimized parallel reduction
__global__ void sumReductionKernel(float *input, float *output, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first add of reduction
    sdata[tid] = 0;
    if (i < N) sdata[tid] = input[i];
    if (i + blockDim.x < N) sdata[tid] += input[i + blockDim.x];
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll last 6 iterations (warp size = 32)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Wrapper function for parallel reduction
float parallelReduceSum(float* d_input, int N, cudaStream_t& stream) {
    const int threadsPerBlock = 256;
    const int blocks = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    
    // Allocate memory for partial sums
    float *d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    
    // First reduction step
    sumReductionKernel<<<blocks, threadsPerBlock, 
                        threadsPerBlock * sizeof(float), stream>>>(
        d_input, d_partial_sums, N
    );
    
    // Final result
    float final_sum;
    
    if (blocks > 1) {
        // Second reduction if needed
        float *d_final_sum;
        cudaMalloc(&d_final_sum, sizeof(float));
        
        sumReductionKernel<<<1, threadsPerBlock, 
                            threadsPerBlock * sizeof(float), stream>>>(
            d_partial_sums, d_final_sum, blocks
        );
        
        cudaMemcpyAsync(&final_sum, d_final_sum, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final_sum);
    } else {
        cudaMemcpyAsync(&final_sum, d_partial_sums, sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaFree(d_partial_sums);
    
    return final_sum;
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 128;
  size_t size = N * sizeof(float);

  // Create CUDA stream for async operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Allocate memory
  float *a, *b, *c;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // Prefetch data to GPU
  cudaMemPrefetchAsync(a, size, deviceId, stream);
  cudaMemPrefetchAsync(b, size, deviceId, stream);
  cudaMemPrefetchAsync(c, size, deviceId, stream);

  // Initialize data
  initWith(2, a, N);
  initWith(1, b, N);
  initWith(0, c, N);

  // Optimize grid and block dimensions
  int threadsPerBlock = 256;
  int blocksPerSM = 32;
  int numberOfBlocks = numberOfSMs * blocksPerSM;
  
  // Ensure we have enough threads to cover all elements
  numberOfBlocks = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

  // Start timing after setup
  cudaStreamSynchronize(stream);
  clock_t start = clock();

  // Launch kernel
  restVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, stream>>>(c, a, b, N);

  // Check for errors
  cudaError_t addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  // Compute final sum using parallel reduction
  float sum = parallelReduceSum(c, N, stream);
  sum = sqrt(sum);

  clock_t end = clock();
  double duration = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;
  printf("Time taken: %.2f microseconds\n", duration);
  printf("Result: %.2f\n", sum);

  // Cleanup
  cudaStreamDestroy(stream);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
