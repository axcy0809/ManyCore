# include <stdio.h>
# include "timer.hpp"
//# include <"random">

__global__ void dot_pro(double *x, double *y, double *tmp, unsigned int N)
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

int main (void)
{
    int N = 10000;
    int s = 256;
    int anz = 10;
    double *px, *py, *d_px, *d_py;
    double *prod, *d_prod;
    double *tmp, *d_tmp;
    double sumdot = 0;
    Timer timer;

    prod = new double[N];
    px = new double[N];
    py = new double[N];
    tmp = new double[s];


    for (int i = 0; i < N; i++)
    {
        px[i] = 1;
        py[i] = 3;
    }
    cudaMalloc(&d_px, N*sizeof(double));
    cudaMalloc(&d_py, N*sizeof(double));
    cudaMalloc(&d_prod, sizeof(double));
    cudaMalloc(&d_tmp, s*sizeof(double));
    cudaMemset(d_prod, 0.0, sizeof(double));

    cudaMemcpy(d_px, px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, N*sizeof(double), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        dot_pro<<<s, s>>>(d_px, d_py, d_tmp, N);
        cudaDeviceSynchronize();
    
        cudaMemcpy(tmp, d_tmp, s*sizeof(double), cudaMemcpyDeviceToHost);

        for(int j = 0; j < s; j++)
        {
            sumdot += tmp[j];
        }
    }
    printf("Time: %g[ms] result: %f\n", (1000*timer.get())/anz,sumdot/anz);
    //printf("FirstEntrieOfSumVec: %f\n",z[N]);


    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_prod);
    free(px);
    free(py);
    free(prod);

    return EXIT_SUCCESS;


}