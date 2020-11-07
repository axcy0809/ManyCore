#include <stdio.h>
#include "timer.hpp"

#include <vector>
#include <iostream>

__global__ void init_kernel(int N)
{
    double *x, *y;

    x = new double [N];
    y = new double [N];

    for (int i = 0; i < N; i++) 
    {
        x[i] = i;
        y[i] = N-i-1;
    }
}


int main(void)
{
    int N = 1000000;
    int M = 1;
    Timer timer;

    cudaDeviceSynchronize();
    timer.reset();

    init_kernel<<<(M+255)/256, 256>>>(N);
    cudaDeviceSynchronize();
    printf("Kernel_init_Time: %g[ms]\n", (1000*timer.get()));

    //32.188[ms]

  
   return EXIT_SUCCESS;
}