#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"
#include <vector>
#include <map>
#include <cmath>


/** @brief Generates the system matrix for a 2D finite difference discretization of the heat equation
 *    -\Delta u = 1
 * on a square domain with homogeneous boundary conditions.
 *
 * Parameters:
 *   - points_per_direction: The number of discretization points in x- and y-direction (square domain)
 *   - csr_rowoffsets, csr_colindices, csr_values: CSR arrays. 'rowoffsets' is the offset aray, 'colindices' holds the 0-based column-indices, 'values' holds the nonzero values.
 */

__global__ void csr_matvec(int N, int *rowoffsets, int *colindices, double *values, double const *x, double *y)
{
    for (int row = blockDim.x * blockIdx.x + threadIdx.x;
        row < N;
        row += gridDim.x * blockDim.x)
    {
      double val = 0;
      for (int jj = rowoffsets[row]; jj < rowoffsets[row+1]; ++jj)
      {
          val += values[jj] * x[colindices[jj]];
      }
      y[row] = val;
    }
}

__global__ void vector_plus_alpha_vector(double *x, double *y, double *z, double alpha, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = thread_id; i < N; i += blockDim.x * gridDim.x)
        z[i] = x[i] + alpha * y[i];
        
}

__global__ void dot_product(double *x, double *y, double *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ double cache[256];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2) 
    {
        __syncthreads();
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
    }

	if(threadIdx.x == 0){
		atomicAdd(dot, cache[0]);
	}
}

/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *x, double *y)
{
    csr_matvec<<<256,256>>>(N, csr_rowoffsets, csr_colindices, csr_values, x, y);

}


/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use with CUDA.
 *  Modify as you see fit.
 */
void conjugate_gradient(size_t N,  // number of unknows
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *rhs,
                        double *solution)
                        //, double *init_guess)   // feel free to add a nonzero initial guess as needed
{

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double *p = (double*)malloc(sizeof(double) * N);
  double *r = (double*)malloc(sizeof(double) * N);
  double *Ap = (double*)malloc(sizeof(double) * N);
  double *pAp = (double*)malloc(sizeof(double));
  double *rr = (double*)malloc(sizeof(double));
  double *h_pAp = (double*)malloc(sizeof(double));
  double *h_rr = (double*)malloc(sizeof(double));
  double *h_rr2 = (double*)malloc(sizeof(double));
  double *h_p = (double*)malloc(sizeof(double) * N);
  double *h_r = (double*)malloc(sizeof(double) * N);

  // line 2: initialize r and p:
  std::copy(rhs, rhs+N, h_p);
  std::copy(rhs, rhs+N, h_r);

  cudaMalloc(&p, N*sizeof(double));
  cudaMalloc(&r, N*sizeof(double));
  cudaMalloc(&Ap, N*sizeof(double));
  cudaMalloc(&pAp, N*sizeof(double));
  cudaMalloc(&rr, N*sizeof(double));

  cudaMemcpy(p, h_p, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(r, h_r, N*sizeof(double), cudaMemcpyHostToDevice);

  int iters = 0;
  while (1) {

    // line 4: A*p:
    csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);
    dot_product<<<256,256>>>(p, Ap, pAp, N);
    dot_product<<<256,256>>>(r, r, rr, N);
    cudaMemcpy(h_pAp, pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rr, rr, sizeof(double), cudaMemcpyDeviceToHost);
    double alpha = h_rr/h_pAp;
    vector_plus_alpha_vector<<<256,256>>>(solution, p, solution, alpha, N);
    vector_plus_alpha_vector<<<256,256>>>(r, Ap, r, -alpha, N);
    dot_product<<<256,256>>>(r, r, rr2, N);
    cudaMemcpy(h_rr2, rr2, sizeof(double), cudaMemcpyDeviceToHost);
    if (h_rr < 1) break;
    double beta = h_rr2 / h_rr;
    vector_plus_alpha_vector<<<256,256>>>(r, p, p, beta, N);




    
    /** YOUR CODE HERE
    *
    * similarly for the other operations
    *
    */

    if (iters > 10000) break;  // solver didn't converge
    ++iters;
  }

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations" << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations." << std::endl;

  cuda_Free(p);
  cuda_Free(r);
  cuda_Free(Ap);
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * 5 * N);
  double *csr_values  = (double*)malloc(sizeof(double) * 5 * N);
  int *h_csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *h_csr_colindices =    (int*)malloc(sizeof(double) * 5 * N);
  double *h_csr_values  = (double*)malloc(sizeof(double) * 5 * N);

  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  //
  // Allocate solution vector and right hand side:
  //
  double *h_solution = (double*)malloc(sizeof(double) * N);
  double *h_rhs      = (double*)malloc(sizeof(double) * N);
  double *solution = (double*)malloc(sizeof(double) * N);
  double *rhs      = (double*)malloc(sizeof(double) * N);
  std::fill(h_rhs, h_rhs + N, 1);

  /**
   *
   * YOUR CODE HERE: Allocate GPU arrays as needed
   *
   **/

  cudaMalloc(&csr_rowoffsets, (N+1)*sizeof(double));
  cudaMalloc(&csr_colindices, N*5*sizeof(double));
  cudaMalloc(&csr_values, N*5*sizeof(double));
  cudaMalloc(&solutlion, N*5*sizeof(double));
  cudaMalloc(&rhs, N*5*sizeof(double));

  cudaMemcpy(rhs, h_rhs, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_rowoffsets, h_csr_rowoffsets, (N+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_colindices, h_csr_colindices, N*5*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_values, h_sr_values, N*5*sizeof(double), cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
  //
  conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);


  cudaMemcpy(h_solution, solution, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csr_rowoffsets, csr_rowoffsets, (N+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csr_colindices, csr_colindices, N*5*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csr_values, sr_values, N*5*sizeof(double), cudaMemcpyDeviceToHost);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, h_csr_rowoffsets, h_csr_colindices, h_csr_values, h_rhs, h_solution);
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;

  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main() {

  solve_system(100); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}