# include <stdio.h>
# include "timer.hpp"
# include <iostream>
# include <numeric>
# include <vector>
#include <fstream>
#include <string>
#include "cublas_v2.h"

__global__ void dot_pro(double *x, double *tmp, int N)
{
    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    __shared__ double cache[256];

    double tmpsum = 0.0;
    while(ind < N)
    {
        tmpsum += x[ind]*x[ind];
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

__global__ void mmsszz(double *x, double *dot, int N)
{

    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    __shared__ double sumofallelements[256];
    __shared__ double einsnorm[256];
    __shared__ double zweisnorm[256];
    __shared__ double maxnorm[256];
    __shared__ double zeros[256];


    double sum = 0;
    double einssum = 0;
    double zweissum = 0;
    double max = 0;
    double highNum = 0;
    double count = 0;

    while(ind < N)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) 
        {
            sum += x[i];
            einssum += std::abs(x[i]);
            zweissum += x[i] * x[i];
            if (x[i] == 0)
            {
                count = count + 1;
            }

            if (std::abs(x[i]) > highNum)
            {
                highNum = std::abs(x[i]);
            }
        }
        max = highNum;
        ind += str;
    }

    sumofallelements[threadIdx.x] = sum;
    einsnorm[threadIdx.x] = einssum;
    zweisnorm[threadIdx.x] = zweissum;
    maxnorm[threadIdx.x] = max;
    zeros[threadIdx.x] = count;

    __syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            sumofallelements[threadIdx.x] += sumofallelements[threadIdx.x + i];
            einsnorm[threadIdx.x] += einsnorm[threadIdx.x + i];
            zweisnorm[threadIdx.x] += zweisnorm[threadIdx.x + i];
            maxnorm[threadIdx.x] += maxnorm[threadIdx.x + i];
            zeros[threadIdx.x] += zeros[threadIdx.x + i];
        }
    }

    if(threadIdx.x == 0)
    {
        double highNum = 0;
        for (int i = 0; i < 256; i++)
        {
            if (std::abs(maxnorm[i]) > highNum)
            {
                highNum = std::abs(maxnorm[i]);
            }
        }
        atomicAdd(dot + 0,sumofallelements[0]);
        atomicAdd(dot + 1,einsnorm[0]);
        atomicAdd(dot + 2,std::sqrt(zweisnorm[0]));
        dot[3] = highNum;
        atomicAdd(dot + 4,zeros[0]);
    }
}

template <typename T>
void toCSV(std::fstream& csv, T* array, int size) 
{
  csv << size;
  for (int i = 0; i < size; ++i) 
  {
    csv << ";" << array[i];
  }
  csv << std::endl;
}


int main (void)
{
    std::vector<int> vec_Ns{100, 1000, 10000, 100000, 1000000,10000000, 100000000};
    double *x, *cuda_x;
    double *result, *cuda_result;
    double *dot, *cuda_dot;
    double *cublas_dot, *cuda_cublas_dot;

 
    Timer timer;
    int anz = 100;
    std::vector<double> ownruntime;
    int shift = 0;


    
    std::fstream csv_times;
    std::string csv_times_name = "shared_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;time_shared;time_dot";
        // to csv file
    csv_times << header << std::endl;

    

    for (int& N : vec_Ns) 
    {   
        //
        // init CUBLAS
        //
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
        const size_t sz = sizeof(double) * (size_t)N;
        //
        // generates a random object:
        //
        srand (time(NULL));
        std::cout << "N: " << N << std::endl;

        //
        // allocate host memory:
        //
        std::cout << "Allocating host arrays..." << std::endl;
        x = (double*)malloc(sizeof(double) * N);
        result = (double*)malloc(sizeof(double) * 7);
    

        for (size_t i=0; i<7; ++i) 
        {
            result[i] = 0;
        }

        for (size_t i=0; i<N; ++i) 
        {
            x[i] = rand() % 10 + 1;;
        }
    
        //
        // allocate device memory
        //
        std::cout << "Allocating CUDA arrays..." << std::endl;
        cudaMalloc((void **)(&cuda_x), sz);
        cudaMalloc((void **)(&cuda_result), 7*sizeof(double));
        cudaMalloc((void **)(&cuda_cublas_dot), sizeof(double));
    

        cudaMemcpy(cuda_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_result, result, 7*sizeof(double), cudaMemcpyHostToDevice);
        


        cudaDeviceSynchronize();
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            mmsszz<<<256, 256>>>(cuda_x, cuda_result, N);
        }
        cudaDeviceSynchronize();
        
        cudaMemcpy(result,cuda_result, 7*sizeof(double), cudaMemcpyDeviceToHost);
        ownruntime.push_back (1000*timer.get()/anz);

        cudaDeviceSynchronize();
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            dot_pro<<<256, 256>>>(cuda_x, cuda_dot, N);
        }
        cudaDeviceSynchronize();
        
        cudaMemcpy(dot,cuda_dot, sizeof(double), cudaMemcpyDeviceToHost);
        ownruntime.push_back (1000*timer.get()/anz);

        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            //cublasSasum(h,N,cuda_x,1,cuda_x,cuda_cublas_dot);
        }
        cudaDeviceSynchronize();
        
        //cudaMemcpy(dot,cuda_dot, sizeof(double), cudaMemcpyDeviceToHost);
        ownruntime.push_back (1000*timer.get()/anz);

    

        

        std::cout << "Time: " << ownruntime[shift] << std::endl;
        std::cout << "Sum of all elements: " << result[0] << std::endl;
        std::cout << "1Norm: " << result[1] << std::endl;
        std::cout << "2Norm: " << result[2] << std::endl;
        std::cout << "MaxNorm: " << result[3] << std::endl;
        std::cout << "nomOfzeros: " << result[4] << std::endl;
        std::cout << " " << std::endl;
        std::cout << "Time DotProd: " << ownruntime[shift + 1] << std::endl;

        std::string sep = ";";
        csv_times << N << sep << ownruntime[shift] << sep << ownruntime[shift + 1] << std::endl;



        cudaFree(cuda_x);
        cudaFree(cuda_result);
        free(x);
        free(result);
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;
        shift = shift + 1;

    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_times_name << std::endl;
    return EXIT_SUCCESS;


}