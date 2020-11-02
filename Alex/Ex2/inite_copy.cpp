# include <stdio.h>
# include "timer.hpp"

int main (void)
{
    int N = 1000000;
    double *x, *y, *d_x, *d_y;
    Timer timer;

    x = new double[N];
    y = new double[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = N-i-1;
    }

    cudaDeviceSynchronize();
    timer.reset();

    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));

    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    printf("Kernel_init_Time: %g[ms]\n", (1000*timer.get()));

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return EXIT_SUCCESS;

    //11.189[ms]
}