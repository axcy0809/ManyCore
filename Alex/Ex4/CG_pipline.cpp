#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
 
// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
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

 
// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];
 
  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }
 
  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }
 
  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}
 
 __global__ void x_plus_a_p(int N, double *x, double *p, double *r, double *Ap, double alpha, double beta)
 {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
   {
     x[i] = x[i] + alpha * p[i];
     r[i] = r[i] - alpha * Ap[i];
     p[i] = r[i] + beta * p[i];
   }
 }

 __global__ void diff_dot_prod(int N, double *Ap, double *p, double *r, double *ApAp, double *pAp, double *rr, int *csr_rowoffsets, int *csr_colindices, double *csr_values)
 {
   __shared__ double cache0[512];
   __shared__ double cache1[512];
   __shared__ double cache2[512];
 
  double tmpsum0 = 0;
  double tmpsum1 = 0;
  double tmpsum2 = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * p[csr_colindices[k]];
    }
    Ap[i] = sum;

    tmpsum0 += Ap[i] * Ap[i];
    tmpsum1 += p[i] * Ap[i];
    tmpsum2 += r[i] * r[i];
  }
 
  cache0[threadIdx.x] = tmpsum0;
  cache1[threadIdx.x] = tmpsum1;
  cache2[threadIdx.x] = tmpsum2;

  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      cache0[threadIdx.x] += cache0[threadIdx.x + k];
      cache1[threadIdx.x] += cache1[threadIdx.x + k];
      cache2[threadIdx.x] += cache2[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) {
    atomicAdd(ApAp, cache0[0]);
    atomicAdd(pAp, cache1[0]);
    atomicAdd(rr, cache2[0]);
  }
 }
 
 
/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with CUDA. Modify as you see fit.
 */
void conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;
 
  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);
 
  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_ApAp, *cuda_rr, *cuda_pAp;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_ApAp, sizeof(double));
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));


 
  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
 
  const double zero = 0;
  double residual_norm_squared = 0;
  cudaMemcpy(cuda_rr, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_rr);
  cudaMemcpy(&residual_norm_squared, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
 
  double initial_residual_squared = residual_norm_squared;

  // line 3
  cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);

  // line 4
  cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_p, cuda_Ap, cuda_pAp);
  cudaMemcpy(&alpha, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
  alpha = residual_norm_squared / alpha;

  // line 5
  cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512,512>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
  cudaMemcpy(&beta, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
  beta = alpha * alpha * beta / residual_norm_squared - 1;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {
 
    // lines 7 to 9:
    x_plus_a_p<<<512,512>>>(N, cuda_solution, cuda_p, cuda_r, cuda_Ap, alpha, beta);
 

    //lines 11 to 12:
    cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_rr, &zero, sizeof(double), cudaMemcpyHostToDevice);
    diff_dot_prod<<<512,512>>>(N, cuda_Ap, cuda_p, cuda_r, cuda_ApAp, cuda_pAp, cuda_rr, csr_rowoffsets, csr_colindices, csr_values);
    cudaMemcpy(&beta, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&alpha, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&residual_norm_squared, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);


    // line convergence check:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }
 
    // line 13:
    alpha = residual_norm_squared / alpha;
 
    // line 14:
    beta = alpha * alpha * beta / residual_norm_squared - 1;
 
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
 
  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_ApAp);
  cudaFree(cuda_rr);
  cudaFree(cuda_pAp);
}
 
/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {
 
  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for
 
  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;
 
  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);
 
  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;
  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
                       csr_values);
 
  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);
 
  //
  // Allocate CUDA-arrays //
  //
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
 
  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
 
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
 
int main() {
  for(int i = 100; i < 1200;i = i + 100)
  {
     solve_system(i); // solves a system with 100*100 unknowns
     std::cout << "i:" << i << std::endl;
     std::cout << "" << std::endl;
     std::cout << "" << std::endl;
  }
 
  return EXIT_SUCCESS;
}