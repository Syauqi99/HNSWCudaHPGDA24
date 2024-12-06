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

void process_distance_vector(float *distances, int N){
    for(int i = 0; i < N; i++){
        sum += distances[i];
    }
    sum = sqrt(sum);
    return sum;
}
