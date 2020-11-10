#include <stdio.h>
# include "timer.hpp"
#include <vector>

__global__ void sumofVectors(double* x, double* y, double* z, int N, int k)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = thread_id; i < N-k; i += blockDim.x * gridDim.x)
    {
        z[i+k] = x[i+k] + y[i+k];
    }
}


int main(void)
{
    int N = 10000000;
    double *x, *y, *z, *d_x, *d_y, *d_z;
    int anz = 10;
    std::vector<double> results;

    Timer timer;

    x = new double [N];
    y = new double [N];
    z = new double [N];

    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
        y[i] = 3;
        z[i] = 0;
    }

    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_z, N*sizeof(double));
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

    
    
    for (int k = 1; k < 64; k++)
    {
        cudaDeviceSynchronize();
        timer.reset();
        
        for (int i = 0; i < anz; i++)
        {
            sumofVectors<<<256, 256>>>(d_x, d_y, d_z, N, k);
        }

        cudaDeviceSynchronize();
        results.push_back(1000*timer.get()/anz);
    }
    for (int i = 1; i < 64; i++)
    {
        printf("%f, ",results[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    delete x;
    delete y;
    delete z;

    return EXIT_SUCCESS;
}

