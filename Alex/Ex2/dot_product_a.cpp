# include <stdio.h>
# include "timer.hpp"
//# include <"random">

__global__ void dot_pro_first(double *x, double *y, double *tmp, unsigned int N)
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
        tmp[blockIdx.x] = cache[0];
    }
}

__global__ void dot_pro_second(double *tmp, double *dot_prd)
{
    for (int i = blockDim.x/2; i > 0; i/=2)
    {
        if(threadIdx.x < i)
        {
            tmp[threadIdx.x] += tmp[threadIdx.x + i];
        }
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        *dot_prd = tmp[0];
    }
}

int main (void)
{
    int N = 10000;
    int s = 256;
    int anz = 10;
    double *px, *py, *d_px, *d_py;
    double *prod, *d_prod, *d_tmp;
    Timer timer;

    prod = new double[N];
    px = new double[N];
    py = new double[N];

    for (int i = 0; i < N; i++)
    {
        px[i] = 1;
        py[i] = 3;
    }
    cudaMalloc(&d_px, N*sizeof(double));
    cudaMalloc(&d_py, N*sizeof(double));
    cudaMalloc(&d_prod, sizeof(double));
    cudaMalloc(&d_tmp, s*sizeof(double));

    cudaMemcpy(d_px, px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, N*sizeof(double), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        dot_pro_first<<<s, s>>>(d_px, d_py, d_tmp, N);
        dot_pro_second<<<1, s>>>(d_tmp, d_prod);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(prod, d_prod, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Time: %g[ms] result: %f\n", (1000*timer.get())/anz,*prod);
    //printf("FirstEntrieOfSumVec: %f\n",z[N]);


    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_prod);
    free(px);
    free(py);
    free(prod);

    return EXIT_SUCCESS;


}