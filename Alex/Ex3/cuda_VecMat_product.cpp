#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"
    
    
/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
/*
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *x, double *y)
{
    double val = 0;
    for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N; row += gridDim.x * blockDim.x)
    {
        val = 0;
        for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row + 1]; jj++)
        {
            val += csr_values[jj] * x[csr_colindices[jj]];
        }
        y[row] = val;
    }
    
}
*/

/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
__global__ void csr_matvec_product(size_t N ,
                        int *csr_rowoffsets , int *csr_colindices , double *csr_values,
                        double *x, double *y)
{
    int  row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < N )
    {
        float  dot_Ax = 0;
        int  row_start = csr_rowoffsets[row];
        int  row_end    = csr_rowoffsets[row +1];

        for (int jj = row_start; jj < row_end; jj++)
        {
            dot_Ax += csr_values[jj] * x[csr_colindices[jj]];
        }
        y[row] += dot_Ax;
    }
}




__global__ void dot_pro(double *x, double *y, double *dot, unsigned int N)
{
    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    __shared__ double cache[256];

    double tmpsum = 0.0;
    while(ind < N)
    {
        tmpsum += x[ind]*y[ind];
        ind += str;
    }

    cache[threadIdx.x] = tmpsum;

    __syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(dot,cache[0]);
    }
}

__global__ void vec_plusmin_alpha_vector(double* x, double*y, double*z, double alpha, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = thread_id; i < N; i += blockDim.x * gridDim.x)
    {
        z[i] = x[i] + alpha * y[i];
    }
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
    
    // line 2: initialize r and p:
    std::copy(rhs, rhs+N, p);
    std::copy(rhs, rhs+N, r);
    
    int iters = 0;
    while (1) {
    
    // line 4: A*p:
    printf("%f, ",Ap[10]);
    csr_matvec_product<<<256, 256>>>(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);
    printf("%f, ",Ap[10]);
    //cudaMemcpy (z, d_z, N∗sizeof(double), cudaMemcpyDeviceToHost);
    
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
    
    free(p);
    free(r);
    free(Ap);
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
    
    //
    // fill CSR matrix with values
    //
    generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);
    
    //
    // Allocate solution vector and right hand side:
    //
    double *solution = (double*)malloc(sizeof(double) * N);
    double *rhs      = (double*)malloc(sizeof(double) * N);
    std::fill(rhs, rhs + N, 1);
    
    /**
     *
     * YOUR CODE HERE: Allocate GPU arrays as needed
     *
     **/
    
    //
    // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
    //
    conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
    
    //
    // Check for convergence:
    //
    double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
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

