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

  // Synchronize and get result
  cudaStreamSynchronize(stream);
  
  // Calculate final result
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += c[i];
  }
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
