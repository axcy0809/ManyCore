# include <stdio.h>
# include "timer.hpp"
#include <vector>


__global__ void dot_pro(int N, double *x, double *y0, double *y1, double *y2, double *y3, double *y4, double *y5, double *y6, double *y7, double *dot)
{

    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    __shared__ double cache0[256];
    __shared__ double cache1[256];
    __shared__ double cache2[256];
    __shared__ double cache3[256];
    __shared__ double cache4[256];
    __shared__ double cache5[256];
    __shared__ double cache6[256];
    __shared__ double cache7[256];

    double tmpsum0 = 0.0;
    double tmpsum1 = 0.0;
    double tmpsum2 = 0.0;
    double tmpsum3 = 0.0;
    double tmpsum4 = 0.0;
    double tmpsum5 = 0.0;
    double tmpsum6 = 0.0;
    double tmpsum7 = 0.0;

    double val = x[0];

    while(ind < N)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) 
        {
            val = x[i];
            tmpsum0 += val * y0[i];
            tmpsum1 += val * y1[i];
            tmpsum2 += val * y2[i];
            tmpsum3 += val * y3[i];
            tmpsum4 += val * y4[i];
            tmpsum5 += val * y5[i];
            tmpsum6 += val * y6[i];
            tmpsum7 += val * y7[i];
        }

        ind += str;
    }

    cache0[threadIdx.x] = tmpsum0;
    cache1[threadIdx.x] = tmpsum1;
    cache2[threadIdx.x] = tmpsum2;
    cache3[threadIdx.x] = tmpsum3;
    cache4[threadIdx.x] = tmpsum4;
    cache5[threadIdx.x] = tmpsum5;
    cache6[threadIdx.x] = tmpsum6;
    cache7[threadIdx.x] = tmpsum7;

    __syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            cache0[threadIdx.x] += cache0[threadIdx.x + i];
            cache1[threadIdx.x] += cache1[threadIdx.x + i];
            cache2[threadIdx.x] += cache2[threadIdx.x + i];
            cache3[threadIdx.x] += cache3[threadIdx.x + i];
            cache4[threadIdx.x] += cache4[threadIdx.x + i];
            cache5[threadIdx.x] += cache5[threadIdx.x + i];
            cache6[threadIdx.x] += cache6[threadIdx.x + i];
            cache7[threadIdx.x] += cache7[threadIdx.x + i];
        }
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(dot + 0,cache0[0]);
        atomicAdd(dot + 1,cache1[0]);
        atomicAdd(dot + 2,cache2[0]);
        atomicAdd(dot + 3,cache3[0]);
        atomicAdd(dot + 4,cache4[0]);
        atomicAdd(dot + 5,cache5[0]);
        atomicAdd(dot + 6,cache6[0]);
        atomicAdd(dot + 7,cache7[0]);
    }
}

int main (void)
{
    int N = 100000;
    int K = 16;
    int s = 256;
    int anz = 10;

    double *x, **y;
    double *d_x, *d_y0, *d_y1, *d_y2, *d_y3, *d_y4, *d_y5, *d_y6, *d_y7;
    double *res_dot, *res_cblas;
    double *d_res_cblas;
    std::vector<double> finalres;

    Timer timer;

    printf("Allocating host arrays...\n");
    x = new double [N];
    y = new double* [K];
    for (int i = 0; i < K; ++i)
        y[i] = new double[N];

    res_dot  = new double [K];
    res_cblas = new double [8];


    printf("Allocating CUDA arrays...\n"); 
    cudaMalloc( (void **)(&d_x), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y0), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y1), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y2), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y3), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y4), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y5), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y6), sizeof(double)*N);
    cudaMalloc( (void **)(&d_y7), sizeof(double)*N);
    cudaMalloc( (void **)(&d_res_cblas), sizeof(double)*8);


    for (int j=0; j<N; ++j) {
      x[j] = 1 + j;
    }
    for (int i=0; i<K; ++i) {
      for (int j=0; j<N; ++j) {
        y[i][j] = 1 + j;
      }
    }

    for (int i=0; i<K; ++i) {
      res_dot[i] = 0;
      res_cblas[i] = 0;
      for (int j=0; j<N; ++j) {
        res_dot[i] += x[j] * y[i][j];
      }
    }


    printf("Copying data to GPU...\n"); 
    cudaMemcpy(d_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y0, y[0], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, y[1], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y[2], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y3, y[3], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y4, y[4], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y5, y[5], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y6, y[6], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y7, y[7], sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_cblas, res_cblas, sizeof(double)*K, cudaMemcpyHostToDevice);

    double *resultslarge  = (double*)malloc(sizeof(double) * K);
    timer.reset();

    for (int g = 0; g < anz; g++)
    {
        for (int i=0; i<K/8; ++i)
        {
            cudaDeviceSynchronize();
            cudaMemcpy(d_y0, y[i*8+0], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y1, y[i*8+1], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y2, y[i*8+2], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y3, y[i*8+3], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y4, y[i*8+4], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y5, y[i*8+5], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y6, y[i*8+6], sizeof(double)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y7, y[i*8+7], sizeof(double)*N, cudaMemcpyHostToDevice);
            dot_pro<<<s, s>>>(N, d_x, d_y0, d_y1, d_y2, d_y3, d_y4, d_y5, d_y6, d_y7, d_res_cblas);
            cudaMemcpy(resultslarge+i*8, d_res_cblas, sizeof(double)*8, cudaMemcpyDeviceToHost);
            
            for (int j = 0; j < 8; j++) 
            {
            res_cblas[j] = 0;
            }
            cudaMemcpy(d_res_cblas, res_cblas, sizeof(double)*8, cudaMemcpyHostToDevice);
      }
    }

    printf("Dot product took %g seconds", 1000*timer.get()/anz);

    free(x);
    cudaFree(d_x);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudaFree(d_y3);
    cudaFree(d_y4);
    cudaFree(d_y5);
    cudaFree(d_y6);
    cudaFree(d_y7);

    //free(res_cblas);
    //free(d_res_cblas);
    free(resultslarge);
    return EXIT_SUCCESS;


}


