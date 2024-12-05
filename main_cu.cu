#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>  // Added this header for std::cout

__global__ void cuda_euclidean_distance(float *a, float *b, float *result, int N) {
    // Declare shared memory array - visible to all threads in the same block
    // Size must be specified when kernel is launched
    extern __shared__ float shared_data[];

    // Calculate global thread ID and stride
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID
    int gridStride = blockDim.x * gridDim.x;               // Total number of threads
    int localId = threadIdx.x;                             // Local thread ID within the block

    // Initialize local sum for this thread
    float sum = 0.0f;

    // Each thread processes multiple elements with grid-stride loop
    // This allows handling arrays larger than total number of threads
    for (int i = threadId; i < N; i += gridStride) {
        float diff = a[i] - b[i];        // Calculate difference
        sum += diff * diff;              // Add squared difference to local sum
    }

    // Store this thread's sum in shared memory
    shared_data[localId] = sum;
    
    // Ensure all threads in block have written to shared memory
    __syncthreads();

    // Parallel reduction in shared memory
    // This loop reduces the partial sums in shared memory to a single sum per block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localId < stride) {
            // Each thread adds a value from the second half to the first half
            shared_data[localId] += shared_data[localId + stride];
        }
        // Ensure all threads have finished reading shared memory before next iteration
        __syncthreads();
    }

    // Only thread 0 in each block writes the final result
    if (localId == 0) {
        // Atomically add this block's sum to the global result
        // atomicAdd is necessary because multiple blocks may write simultaneously
        atomicAdd(result, shared_data[0]);
    }
}

int main() {
  int N = 1000000;  // Size of arrays

  
  // Allocate host memory
  h_a = (float*)malloc(N * sizeof(float));
  h_b = (float*)malloc(N * sizeof(float));
  h_result = (float*)malloc(sizeof(float));

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
      h_a[i] = static_cast<float>(i);
      h_b[i] = static_cast<float>(i + 1);
  }
  *h_result = 0.0f;



  // Start timing
  auto start = std::chrono::high_resolution_clock::now();
  // Host code to launch the kernel
  float *d_a, *d_b, *d_result;
  size_t size = N * sizeof(float);

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_result, sizeof(float));

  if (checkCuda(cudaMallocManaged(&d_a, size)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaMallocManaged(&d_b, size)) != cudaSuccess){
    exit(1);
  }

  if (checkCuda(cudaMallocManaged(&d_c, size)) != cudaSuccess){
    exit(1);
  }

  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  threadsPerBlock = props.maxThreadsPerBlock;
  numberOfBlocks = props.multiProcessorCount * 2;

  cudaMemPrefetchAsync(d_a, size, deviceId);
  cudaMemPrefetchAsync(d_b, size, deviceId);

  // Initialize result to 0
  cudaMemset(d_result, 0, sizeof(float));

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  // Launch kernel with shared memory size = blockSize * sizeof(float)
  cuda_euclidean_distance<<<numberOfBlocks, threadsPerBlock>>>(
      d_a, d_b, d_result, N
  );

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // End timing
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Time taken: " << duration << " microseconds" << std::endl;

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  cudaMemPrefetchAsync(d_result, sizeof(float), cudaCpuDeviceId); // Prefetch c to CPU

  return 0;
}
