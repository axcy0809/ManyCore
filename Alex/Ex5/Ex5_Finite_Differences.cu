#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <fstream>

/////////////////////////////// GD functions from the 4.Exercise /////////////////////////////////////////////////////////////////////////////
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
    int *csr_colindices, double *csr_values,
    double *x, double *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) 
    {
        double sum = 0;
        for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) 
        {
            sum += csr_values[k] * x[csr_colindices[k]];
        }
        y[i] = sum;
    }
}


// x <- x + alpha * y
__global__ void cuda_vecadd(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] += alpha * y[i];
}

// x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] = y[i] + alpha * x[i];
}

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) 
  {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) 
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}


/////////////////////////////// Setup for exclusive scan /////////////////////////////////////////////////////////////////////////////
 
__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ int shared_buffer[512];
  int my_value;
 
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
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[512];

 
  // load data:
  int my_carry = carries[threadIdx.x];
 
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
 
__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
 
  __shared__ int shared_offset;
 
  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];
 
  __syncthreads();
 
  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

/////////////////////////////// nonzero count function /////////////////////////////////////////////////////////////////////////////

__global__ void count_nonzero_entries(int* row_offsets, int N, int M) {
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

/////////////////////////////// GPU assemble /////////////////////////////////////////////////////////////////////////////

__global__ void assemble_A_GPU(double* values, int* columns, int* row_offsets, int N, int M) {
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

/////////////////////////////// CPU assemble /////////////////////////////////////////////////////////////////////////////

void assemble_A_CPU(double* values, int* columns, int* row_offsets, int N, int M) {
  for(int row = 0; row < N*M; row++) {
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

/////////////////////////////// given exclusive scan /////////////////////////////////////////////////////////////////////////////
 
void exclusive_scan(int const * input,
                    int       * output, int N)
{
  int num_blocks = 512;
  int threads_per_block = 512;
 
  int *carries;
  cudaMalloc(&carries, sizeof(int) * num_blocks);
 
  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);
 
  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, num_blocks>>>(carries);
 
  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);
 
  cudaFree(carries);
}
 

 /////////////////////////////// GD functions from the 4.Exercise /////////////////////////////////////////////////////////////////////////////
 
 void conjugate_gradient(int N, // number of unknows
  int *csr_rowoffsets, int *csr_colindices,
  double *csr_values, double *rhs, double *solution, int max)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_scalar;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_scalar, sizeof(double));

  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  const double zero = 0;
  double residual_norm_squared = 0;
  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_scalar);
  cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  double initial_residual_squared = residual_norm_squared;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

  // line 4: A*p:
  cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);

  // lines 5,6:
  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_p, cuda_Ap, cuda_scalar);
  cudaMemcpy(&alpha, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
  alpha = residual_norm_squared / alpha;

  // line 7:
  cuda_vecadd<<<512, 512>>>(N, cuda_solution, cuda_p, alpha);

  // line 8:
  cuda_vecadd<<<512, 512>>>(N, cuda_r, cuda_Ap, -alpha);

  // line 9:
  beta = residual_norm_squared;
  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_scalar);
  cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  // line 10:
  if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
  break;
  }

  // line 11:
  beta = residual_norm_squared / beta;

  // line 12:
  cuda_vecadd2<<<512, 512>>>(N, cuda_p, cuda_r, beta);

  if (iters > 10000)
  break; // solver didn't converge
  ++iters;
  }
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  if (iters > 10000)
  std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
  << std::endl;
  else
  std::cout << "Conjugate Gradient converged in " << iters << " iterations."
  << std::endl;

  /////////////////////////////// ineffizient output only for the bonuspoint part /////////////////////////////////////////////////////////////////////////////

/*
  int ck = 0; 
  for (int i = 0; i < N/max; i++)
  {
    for (int j = 0; j < N/max; j++)
    {
      std::cout << solution[ck] << "," << std::endl;
      ck = ck + 1;
    }
    std::cout << " " << std::endl;  
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
  }
  */



  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_scalar);
}

 /////////////////////////////// GD functions from the 4.Exercise /////////////////////////////////////////////////////////////////////////////

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
 void solve_system(int x_dim, int y_dim, int* csr_rowoffsets, double* csr_values, int* csr_colindices, int max) {

  int N = x_dim * y_dim;
  int *cuda_csr_rowoffsets, *cuda_csr_colindices; 
  double *cuda_csr_values;

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  //
  // Allocate CUDA-arrays //
  //
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(int) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution,max);

  //
  // Check for convergence:
  //

  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;

  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}

template <typename T>
void printContainer(T container, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << container[i] << " | ";
    }
}

 
int main() 
{
  int N = 3;
  int M = N; //quadratic matrix

  Timer timer;

  //
  // Allocate host arrays for reference
  //
  int *row_offsets = (int *)malloc(sizeof(int) * (N*M+1));


  //
  // Allocate CUDA-arrays
  //
  int *cuda_row_offsets;
  int *cuda_row_offsets_2;
  double *cuda_values;
  int *cuda_columns;

  cudaMalloc(&cuda_row_offsets, sizeof(int) * (N*M+1));
  cudaMalloc(&cuda_row_offsets_2, sizeof(int) * (N*M+1));


  // Perform the calculations
  count_nonzero_entries<<<512, 512>>>(cuda_row_offsets, N, M);
  exclusive_scan(cuda_row_offsets, cuda_row_offsets_2, N*M+1);
  cudaMemcpy(row_offsets, cuda_row_offsets_2, sizeof(int) * (N*M+1), cudaMemcpyDeviceToHost);

  int numberOfValues = (int)row_offsets[N*M];
  double *values = (double *)malloc(sizeof(double) * numberOfValues);
  int *columns = (int *)malloc(sizeof(int) * numberOfValues);
  cudaMalloc(&cuda_values, sizeof(double) * numberOfValues);
  cudaMalloc(&cuda_columns, sizeof(int) * numberOfValues);

   /////////////////////////////// Benchmark the time GPU /////////////////////////////////////////////////////////////////////////////

  timer.reset();
  assemble_A_GPU<<<512, 512>>>(cuda_values, cuda_columns, cuda_row_offsets_2, N, M);
  std::cout << "Time to assemble on GPU: " << timer.get() <<  std::endl;
  cudaMemcpy(values, cuda_values, sizeof(double) * numberOfValues, cudaMemcpyDeviceToHost);
  cudaMemcpy(columns, cuda_columns, sizeof(int) * numberOfValues, cudaMemcpyDeviceToHost);

  /////////////////////////////// Benchmark the time CPU /////////////////////////////////////////////////////////////////////////////

  //timer.reset();
  //assemble_A_CPU(values, columns, row_offsets, N, M);
  //std::cout << "Time to assemble on CPU: " << timer.get() <<  std::endl;

  printContainer(values, numberOfValues);
  std::cout << std::endl;
  printContainer(columns, numberOfValues);

  solve_system(N, M, row_offsets, values, columns, N);


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