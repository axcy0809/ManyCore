#include <iostream>
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

// y = A * x
__global__ void sparseVector(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;

    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }

    y[i] = sum;
  }
}


// Y = A * X
__global__ void sparseDenseRowMajor(int N, int K, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

    for (int k = 0; k < K; k++) {
      double sum = 0;

      for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
        sum += csr_values[jj] * x[csr_colindices[jj]*K + k];
      }

      y[i + N*k] = sum;
    }

  }
}


// Y = A * X
__global__ void sparseDenseColumnMajor(int N, int K, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

    for (int k = 0; k < K; k++) {
      double sum = 0;

      for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
        sum += csr_values[jj] * x[csr_colindices[jj] + N*k];
      }

      y[i + N*k] = sum;
    }

  }
}


template <typename T>
 void printContainer(T container,  int N) {
     for (int i = 0; i < N; i++) {
         std::cout << container[i] << " | ";
    }
    std::cout << std::endl;
}


int main() {

  Timer tim1, tim2, tim3;
  int N = 5;
  int K = 3;

  double *values = ( double *)malloc(sizeof( double) * N*10);
  int *colindices = (int*)malloc(sizeof(int) * N*10);
  int *offsets = (int*)malloc(sizeof(int) * (N+1));

  double *x = ( double *)malloc(sizeof( double) * N);
  double *X = (double*)malloc(sizeof(double) * N*K);

  double *y = ( double *)malloc(sizeof( double) * N);
  double *Y = (double*)malloc(sizeof(double) * N*K);

  values[0] = 2;
  values[1] = 7;
  values[2] = 4;
  values[3] = 1;
  values[4] = 9;
  values[5] = 3;
  values[6] = 6;
  values[7] = 5;
  values[8] = 8;

  colindices[0] = 1;
  colindices[1] = 3;
  colindices[2] = 4;
  colindices[3] = 2;
  colindices[4] = 3;
  colindices[5] = 0;
  colindices[6] = 2;
  colindices[7] = 4;
  colindices[8] = 1;

  offsets[0] = 0;
  offsets[1] = 3;
  offsets[2] = 5;
  offsets[3] = 6;
  offsets[4] = 8;
  offsets[5] = 9;

  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;
  x[4] = 5;

  X[0] = 1;
  X[1] = 1;
  X[2] = 1;
  X[3] = 2;
  X[4] = 2;
  X[5] = 2;
  X[6] = 3;
  X[7] = 3;
  X[8] = 3;
  X[9] = 4;
  X[10] = 4;
  X[11] = 4;
  X[12] = 5;
  X[13] = 5;
  X[14] = 5;

  double    *cuda_values;
  int       *cuda_colindices;
  int       *cuda_offsets;
  double    *cuda_x;
  double    *cuda_X;
  double    *cuda_y;
  double    *cuda_Y;

  cudaMalloc(&cuda_values, sizeof( double) * N*10);
  cudaMalloc(&cuda_colindices, sizeof( int) * N*10);
  cudaMalloc(&cuda_offsets, sizeof( int) * (N+1));
  cudaMalloc(&cuda_x, sizeof( double) * N);
  cudaMalloc(&cuda_X, sizeof( double) * N*K);
  cudaMalloc(&cuda_y, sizeof( double) * N);
  cudaMalloc(&cuda_Y, sizeof( double) * N*K);

  cudaMemcpy(cuda_values, values, sizeof( double) * N*10, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_colindices, colindices, sizeof( int) * N*10, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_offsets, offsets, sizeof( int) * (N+1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_x, x, sizeof( double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_X, X, sizeof( double) * N*K, cudaMemcpyHostToDevice);

  std::vector<double> timings1;
    for(int reps=0; reps < 10; ++reps) {
        tim1.reset();
        for (int j = 0; j < K; j++) {
          sparseVector<<<256, 256>>>(N, cuda_offsets, cuda_colindices, cuda_values, cuda_x, cuda_y);
          cudaMemcpy(y, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);
        }
        timings1.push_back(tim1.get());
    }
  std::sort(timings1.begin(), timings1.end());
  double time_elapsed1 = timings1[10/2];
  std::cout << "Time elapsed vector: " << time_elapsed1 << std::endl << std::endl;






  std::vector<double> timings2;
    for(int reps=0; reps < 10; ++reps) {
        tim2.reset();
        sparseDenseRowMajor<<<256, 256>>>(N, K, cuda_offsets, cuda_colindices, cuda_values, cuda_X, cuda_Y);
        cudaMemcpy(Y, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
        timings2.push_back(tim2.get());
    }
  std::sort(timings2.begin(), timings2.end());
  double time_elapsed2 = timings2[10/2];
  std::cout << "Time elapsed row: " << time_elapsed2 << std::endl << std::endl;





  std::vector<double> timings3;
  for(int reps=0; reps < 10; ++reps) {
      tim3.reset();
      sparseDenseRowMajor<<<256, 256>>>(N, K, cuda_offsets, cuda_colindices, cuda_values, cuda_X, cuda_Y);
      cudaMemcpy(Y, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
      timings3.push_back(tim3.get());
  }
  std::sort(timings3.begin(), timings3.end());
  double time_elapsed3 = timings3[10/2];
  std::cout << "Time elapsed column: " << time_elapsed3 << std::endl << std::endl;
  

  

  free(x);
  free(X);
  free(y);
  free(Y);
  free(values);
  free(colindices);
  free(offsets);

  cudaFree(cuda_values);
  cudaFree(cuda_colindices);
  cudaFree(cuda_offsets);
  cudaFree(cuda_x);
  cudaFree(cuda_X);
  cudaFree(cuda_y);
  cudaFree(cuda_Y);


  return EXIT_SUCCESS;
}
