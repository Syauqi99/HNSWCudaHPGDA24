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
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] - b[i];
    result[i] = result[i] * result[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 128;
  size_t size = N * sizeof(float);

  clock_t start = clock();

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(2, a, N);
  initWith(1, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = N;
  numberOfBlocks = 3;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  restVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // sum the result
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += c[i];
  }
  sum = sqrt(sum);
  printf("Sum of result: %f\n", sum);

  clock_t end = clock();
  double duration = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;
  printf("Time taken: %.2f microseconds\n", duration);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
