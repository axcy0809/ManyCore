#include <stdio.h>
#include "timer.hpp"


int main(void)
{
    //N = [10,100,1000,10000,100000,1000000,10000000]
    int N = 10;

    double *d_x;
    Timer timer;


    timer.reset();
    for (int i = 0;i < 100; i++)
    {
        cudaMalloc(&d_x, N*sizeof(double));
        cudaFree(d_x);
    }
    printf("Malloc_Free_Time: %g[ms] N = %d\n", (1000*timer.get())/100,N);

   return EXIT_SUCCESS;
}