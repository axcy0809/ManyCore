#include <stdio.h>
#include "timer.hpp"
#include <iostream>
#include <algorithm>
#include <vector>


#define MAGIC_NUMBER 10

__global__ void sumVectors(double *x, double *y, double *z, int N, int k)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = thread_id; i < N/k; i += blockDim.x * gridDim.x)
        z[k*i] = x[k*i] + y[k*i];
        
}

int main(void)
{
    int N = 100000000;
    int k = 20;
    
    double*x, *y, *z, *d_x, *d_y, *d_z;
    Timer timer;

    x = new double[N];
    y = new double[N];
    z = new double[N];


    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
        y[i] = 2;
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
    std::vector<double> timings;

    for(int reps=0; reps < MAGIC_NUMBER; ++reps) 
    {
        sumVectors<<<256, 256>>>(d_x, d_y, d_z, N, k);
        cudaDeviceSynchronize();
        timings.push_back(timer.get());
    }
    
    std::sort(timings.begin(), timings.end());
    double time_elapsed = timings[MAGIC_NUMBER/2];

    cudaMemcpy(z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    printf("Addition took %g seconds", time_elapsed);

    std::cout << std::endl << "z[0] = " << z[0] << std::endl;
    std::cout << "z[1] = " << z[1] << std::endl;
    std::cout << "z[k] = " << z[k] << std::endl;
    std::cout << "z[2*k] = " << z[2*k-1] << std::endl;
    std::cout << "z[2*k+1] = " << z[2*k-1+1] << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    delete x;
    delete y;
    delete z;

    return EXIT_SUCCESS;
}
////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "timer.hpp"
#include <iostream>
#include <algorithm>
#include <vector>


#define MAGIC_NUMBER 10

__global__ void sumVectors(double *x, double *y, double *z, int N, int k)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = thread_id; i < N/k; i += blockDim.x * gridDim.x)
        z[k*i] = x[k*i] + y[k*i];
        
}

int main(void)
{
    int N = 100000000;
    
    double*x, *y, *z, *d_x, *d_y, *d_z;
    Timer timer;

    x = new double[N];
    y = new double[N];
    z = new double[N];


    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
        y[i] = 2;
        z[i] = 0;
    }

    
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_z, N*sizeof(double));
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

    for(int k = 1; k < 64; k += 2) 
    {
        cudaDeviceSynchronize();
       
        std::vector<double> timings;

        for(int reps=0; reps < MAGIC_NUMBER; ++reps) 
        {
             timer.reset();
            sumVectors<<<256, 256>>>(d_x, d_y, d_z, N, k);
            cudaDeviceSynchronize();
            timings.push_back(timer.get());
        }
        
        std::sort(timings.begin(), timings.end());
        double time_elapsed = timings[MAGIC_NUMBER/2];

        std::cout << time_elapsed << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    delete x;
    delete y;
    delete z;

    return EXIT_SUCCESS;
}

