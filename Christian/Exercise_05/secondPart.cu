#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
 
 
 
 
__global__ void scan_kernel_1(double const *X,
                              double *Y,
                              int N,
                              double *carries)
{
  __shared__ double shared_buffer[256];
  double my_value;
 
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;
 
  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;
 
    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();
 
    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
 
    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;
 
    block_offset += shared_buffer[blockDim.x-1];
  }
 
  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;
 
}
 
// exclusive-scan of carries
__global__ void scan_kernel_2(double *carries)
{
  __shared__ double shared_buffer[256];
 
  // load data:
  double my_carry = carries[threadIdx.x];
 
  // exclusive scan in shared buffer:
 
  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();
 
  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}
 
__global__ void scan_kernel_3(double *Y, int N,
                              double const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
 
  __shared__ double shared_offset;
 
  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];
 
  __syncthreads();
 
  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

__global__ void count_nnz(double* row_offsets, int N, int M) {
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N * M; row += gridDim.x * blockDim.x) {
        int nnz_for_this_node = 1;
        int i = row / N;
        int j = row % N;

        if(i > 0) nnz_for_this_node += 1;
        if(j > 0) nnz_for_this_node += 1;
        if(i < N-1) nnz_for_this_node += 1;
        if(j < M-1) nnz_for_this_node += 1;
        
        row_offsets[row] = nnz_for_this_node;
    }
}


__global__ void populate_values(double* values, int* columns, double* row_offsets, int N, int M) {
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) {
        int i = row / N;
        int j = row % N;
        int counter = 0;

        if ( i > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i-1)*N+j;
            counter++;
        }
        
        if ( j > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j-1);
            counter++;
        }

        values[(int)row_offsets[row] + counter] = 4;
        columns[(int)row_offsets[row] + counter] = i*N+j;

        counter++;

        if ( j < M-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j+1);
            counter++;
        }
        if ( i < N-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i+1)*N+j;
            counter++;
        }
    }
}

 
void exclusive_scan(double const * input,
                    double       * output, int N)
{
  int num_blocks = 256;
  int threads_per_block = 256;
 
  double *carries;
  cudaMalloc(&carries, sizeof(double) * num_blocks);
 
  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);
 
  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, num_blocks>>>(carries);
 
  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);
 
  cudaFree(carries);
}
 
 
 
 template <typename T>
 void printContainer(T container, int N) {
     for (int i = 0; i < N; i++) {
         std::cout << container[i] << " | ";
     }
 }
 
 
int main() {
 
  int N = 3;
  int M = 3;
 
  //
  // Allocate host arrays for reference
  //
  double *row_offsets = (double *)malloc(sizeof(double) * (N*M+1));
  
 
  //
  // Allocate CUDA-arrays
  //
  double *cuda_row_offsets;
  double *cuda_row_offsets_2;
  double *cuda_values;
  int *cuda_columns;

  cudaMalloc(&cuda_row_offsets, sizeof(double) * (N*M+1));
  cudaMalloc(&cuda_row_offsets_2, sizeof(double) * (N*M+1));
 
 
  // Perform the calculations
  count_nnz<<<256, 256>>>(cuda_row_offsets, N, M);
  exclusive_scan(cuda_row_offsets, cuda_row_offsets_2, N*M+1);
  cudaMemcpy(row_offsets, cuda_row_offsets_2, sizeof(double) * (N*M+1), cudaMemcpyDeviceToHost);

  printContainer(row_offsets, N*M+1);
  std::cout << std::endl;


  int numberOfValues = (int)row_offsets[N*M];
  double *values = (double *)malloc(sizeof(double) * numberOfValues);
  int *columns = (int *)malloc(sizeof(int) * numberOfValues);
  cudaMalloc(&cuda_values, sizeof(double) * numberOfValues);
  cudaMalloc(&cuda_columns, sizeof(int) * numberOfValues);

  populate_values<<<256, 256>>>(cuda_values, cuda_columns, cuda_row_offsets_2, N, M);
  cudaMemcpy(values, cuda_values, sizeof(double) * numberOfValues, cudaMemcpyDeviceToHost);
  cudaMemcpy(columns, cuda_columns, sizeof(int) * numberOfValues, cudaMemcpyDeviceToHost);



  printContainer(values, numberOfValues);
  std::cout << std::endl;
  printContainer(columns, numberOfValues);
 
 
  //
  // Clean up:
  //
  free(row_offsets);
  free(values);
  free(columns);
  cudaFree(cuda_row_offsets);
  cudaFree(cuda_row_offsets_2);
  cudaFree(cuda_values);
  cudaFree(cuda_columns);
  return EXIT_SUCCESS;
}