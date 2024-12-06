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

// Single-kernel reduction optimized for small arrays (N <= 1024)
__global__ void smallArrayReductionKernel(float *input, float *output, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    
    // Load input into shared memory
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();
    
    // Unrolled reduction for better performance
    if (N >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (N >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (N >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); }
    
    // Last warp reduction (no sync needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (N >= 64) smem[tid] += smem[tid + 32];
        if (N >= 32) smem[tid] += smem[tid + 16];
        if (N >= 16) smem[tid] += smem[tid + 8];
        if (N >= 8)  smem[tid] += smem[tid + 4];
        if (N >= 4)  smem[tid] += smem[tid + 2];
        if (N >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) output[0] = sdata[0];
}

// Simple wrapper for parallel reduction
float parallelReduceSum(float* d_input, int N, cudaStream_t& stream) {
    float final_sum;
    float *d_output;
    cudaMalloc(&d_output, sizeof(float));
    
    // Round up to nearest warp size (32)
    int threadsNeeded = (N + 31) / 32 * 32;
    
    // Launch single kernel for reduction
    smallArrayReductionKernel<<<1, threadsNeeded, threadsNeeded * sizeof(float), stream>>>(
        d_input, d_output, N
    );
    
    // Copy result back to host
    cudaMemcpyAsync(&final_sum, d_output, sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaFree(d_output);
    
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
