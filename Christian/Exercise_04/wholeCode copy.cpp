#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
 
 __global__ void cuda_many_dot_product(int N, double *x, double *y0, double *y1, double *y2, double *y3, double *y4, double *y5, double *y6, double *y7, double *result)
{
  __shared__ double shared_mem_0[512];
  __shared__ double shared_mem_1[512];
  __shared__ double shared_mem_2[512];
  __shared__ double shared_mem_3[512];
  __shared__ double shared_mem_4[512];
  __shared__ double shared_mem_5[512];
  __shared__ double shared_mem_6[512];
  __shared__ double shared_mem_7[512];
 
  double dot_0 = 0;
  double dot_1 = 0;
  double dot_2 = 0;
  double dot_3 = 0;
  double dot_4 = 0;
  double dot_5 = 0;
  double dot_6 = 0;
  double dot_7 = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double val = x[i];
    dot_0 += val * y0[i];
    dot_1 += val * y1[i];;
    dot_2 += val * y2[i];;
    dot_3 += val * y3[i];;
    dot_4 += val * y4[i];;
    dot_5 += val * y5[i];;
    dot_6 += val * y6[i];;
    dot_7 += val * y7[i];;
  }
 
  shared_mem_0[threadIdx.x] = dot_0;
  shared_mem_1[threadIdx.x] = dot_1;
  shared_mem_2[threadIdx.x] = dot_2;
  shared_mem_3[threadIdx.x] = dot_3;
  shared_mem_4[threadIdx.x] = dot_4;
  shared_mem_5[threadIdx.x] = dot_5;
  shared_mem_6[threadIdx.x] = dot_6;
  shared_mem_7[threadIdx.x] = dot_7;

  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem_0[threadIdx.x] += shared_mem_0[threadIdx.x + k];
      shared_mem_1[threadIdx.x] += shared_mem_1[threadIdx.x + k];
      shared_mem_2[threadIdx.x] += shared_mem_2[threadIdx.x + k];
      shared_mem_3[threadIdx.x] += shared_mem_3[threadIdx.x + k];
      shared_mem_4[threadIdx.x] += shared_mem_4[threadIdx.x + k];
      shared_mem_5[threadIdx.x] += shared_mem_5[threadIdx.x + k];
      shared_mem_6[threadIdx.x] += shared_mem_6[threadIdx.x + k];
      shared_mem_7[threadIdx.x] += shared_mem_7[threadIdx.x + k];
    }
  }
 
  if (threadIdx.x == 0){
       atomicAdd(result+0, shared_mem_0[0]);
       atomicAdd(result+1, shared_mem_1[0]);
       atomicAdd(result+2, shared_mem_2[0]);
       atomicAdd(result+3, shared_mem_3[0]);
       atomicAdd(result+4, shared_mem_4[0]);
       atomicAdd(result+5, shared_mem_5[0]);
       atomicAdd(result+6, shared_mem_6[0]);
       atomicAdd(result+7, shared_mem_7[0]);
  }
}

int main(void)
{
    const size_t N = 100000;
    const size_t K = 8;
 
    //
    // Initialize CUBLAS:
    //
    std::cout << "Init CUBLAS..." << std::endl;
    cublasHandle_t h;
    cublasCreate(&h);
 
 
    //
    // allocate host memory:
    //
    std::cout << "Allocating host arrays..." << std::endl;
    double  *x = (double*)malloc(sizeof(double) * N);
    double **y = (double**)malloc(sizeof(double*) * K);
    for (size_t i=0; i<K; ++i) {
      y[i] = (double*)malloc(sizeof(double) * N);
    }
    double *results  = (double*)malloc(sizeof(double) * K);
    double *results2 = (double*)malloc(sizeof(double) * K);
 
 
    //
    // allocate device memory
    //
    std::cout << "Allocating CUDA arrays..." << std::endl;
    double *cuda_x; cudaMalloc( (void **)(&cuda_x), sizeof(double)*N);
    double *cuda_y0; cudaMalloc( (void **)(&cuda_y0), sizeof(double)*N);
    double *cuda_y1; cudaMalloc( (void **)(&cuda_y1), sizeof(double)*N);
    double *cuda_y2; cudaMalloc( (void **)(&cuda_y2), sizeof(double)*N);
    double *cuda_y3; cudaMalloc( (void **)(&cuda_y3), sizeof(double)*N);
    double *cuda_y4; cudaMalloc( (void **)(&cuda_y4), sizeof(double)*N);
    double *cuda_y5; cudaMalloc( (void **)(&cuda_y5), sizeof(double)*N);
    double *cuda_y6; cudaMalloc( (void **)(&cuda_y6), sizeof(double)*N);
    double *cuda_y7; cudaMalloc( (void **)(&cuda_y7), sizeof(double)*N);
    double *cuda_results2; cudaMalloc( (void **)(&cuda_results2), sizeof(double)*K);

   
 
    //
    // fill host arrays with values
    //
    for (size_t j=0; j<N; ++j) {
      x[j] = 1 + j%K;
    }
    for (size_t i=0; i<K; ++i) {
      for (size_t j=0; j<N; ++j) {
        y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
      }
    }
 
    //
    // Reference calculation on CPU:
    //
    for (size_t i=0; i<K; ++i) {
      results[i] = 0;
      results2[i] = 0;
      for (size_t j=0; j<N; ++j) {
        results[i] += x[j] * y[i][j];
      }
    }    
   
    //
    // Copy data to GPU
    //
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y0, y[0], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y1, y[1], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y2, y[2], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y3, y[3], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y4, y[4], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y5, y[5], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y6, y[6], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y7, y[7], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_results2, results2, sizeof(double)*K, cudaMemcpyHostToDevice);
    //for (size_t i=0; i<K; ++i) {
    //  cudaMemcpy(cuda_y[i*N], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
    //}
 
 
    //
    // Let CUBLAS do the work:
    //
    std::cout << "Running dot products with CUBLAS..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      //cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
    }
    cudaMemcpy(cuda_y0, y[0], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y1, y[1], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y2, y[2], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y3, y[3], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y4, y[4], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y5, y[5], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y6, y[6], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y7, y[7], sizeof(double)*N, cudaMemcpyHostToDevice);
    cuda_many_dot_product<<<512, 512>>>(N, cuda_x, cuda_y0, cuda_y1, cuda_y2, cuda_y3, cuda_y4, cuda_y5, cuda_y6, cuda_y7, cuda_results2);

    // Get back the results

    cudaMemcpy(results2, cuda_results2, sizeof(double)*K, cudaMemcpyDeviceToHost);
 
    //
    // Compare results
    //
    std::cout << "Copying results back to host..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      std::cout << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
    }
 
    
    //
    // Clean up:
    //
    std::cout << "Cleaning up..." << std::endl;
    free(x);
    cudaFree(cuda_x);
 
    for (size_t i=0; i<K; ++i) {
      free(y[i]);
      //cudaFree(cuda_y[i]);
    }
    free(y);
    //free(cuda_y);
 
    free(results);
    free(results2);
 
    cublasDestroy(h);
    return 0;
}