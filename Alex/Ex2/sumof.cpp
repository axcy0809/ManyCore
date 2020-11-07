# include <stdio.h>
# include "timer.hpp"

__global__ void SumOfVectors(double *x, double *y, double *z, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = thread_id; i < N; i += blockDim.x * gridDim.x)
    {
        z[i] = x[i] + y[i];
    }
}

int main (void)
{
    int N = 10000000;
    int s = 16;
    int anz = 10;
    double *x, *y, *z, *d_x, *d_y, *d_z;
    Timer timer;

    x = new double[N];
    y = new double[N];
    z = new double[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = N-i-1;
        z[i] = 0;
    }

    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_z, N*sizeof(double));

    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        SumOfVectors<<<s, s>>>(d_x, d_y, d_z, N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    printf("SumTime: %g[ms]\n", (1000*timer.get())/anz);
    printf("FirstEntrieOfSumVec: %f\n",z[N]);


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);

    return EXIT_SUCCESS;


}