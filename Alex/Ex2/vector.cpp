#include <stdio.h>
#include "timer.hpp"

__global__
void sum2vec(int n, double *x, double *y)
{
    double sum;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) sum = x[i] + y[i];
}

int main(void)
{
    int N = 1000000;

    double *x, *y, *d_x, *d_y;
    Timer timer;

    // Allocate host memory and initialize
    timer.reset();
    x = (double*)malloc(N*sizeof(double));
    y = (double*)malloc(N*sizeof(double));

    for (int i = 0; i < N; i++) 
    {
        x[i] = i;
        y[i] = N-i;
    }
    printf("b) direct initialisationtime: %g[ms]\n", 1000*timer.get());
    timer.reset();
    // Allocate device memory and copy host data over
    cudaMalloc(&d_x, N*sizeof(double)); 
    cudaMalloc(&d_y, N*sizeof(double));
    printf("a) cudaMalloc_initTime: %g[ms] N = %d\n", 1000*timer.get(),N);


    



    //cudaDeviceSynchronize();
    //sum2vec<<<(N+255)/256, 256>>>(N, d_x, d_y);
 //   timer.reset();
    // Allocate device memory and copy host data over
 //   cudaFree(&d_x, N*sizeof(double)); 
 //   cudaFree(&d_y, N*sizeof(double));
 //   printf("cudaFree_initTime: %g\n", timer.get());
    cudaDeviceSynchronize();
    timer.reset();
    cudaFree(d_x);
    cudaFree(d_y);
    printf("a) cudaFree_initTime: %g[ms] N = %d\n", 1000*timer.get(),N);
    free(x);
    free(y);
   return EXIT_SUCCESS;
}