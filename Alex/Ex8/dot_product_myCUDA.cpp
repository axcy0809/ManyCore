#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

typedef double       ScalarType;

__global__ void dot_product(ScalarType* x, ScalarType* y, ScalarType* dot, int N) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ ScalarType cache[256];

    ScalarType temp = 0;
    while (index < N) {
        temp += (x[index] + y[index]) * (x[index] - y[index]);
        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    for (int i = blockDim.x/2; i > 0; i/= 2) {
        __syncthreads();
        if (threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
    }

    if (threadIdx.x == 0)
        atomicAdd(dot, cache[0]);
}


int main() 
{
    std::fstream csv_times;
    std::string csv_times_name = "dot_myCUDA_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;myCUDA_time";
        // to csv file
    csv_times << header << std::endl;


    Timer timer;
    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 1;
    ScalarType myCUDA_Time = 0;


    for (int i = 1; i < 8; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {
        ScalarType *x = (ScalarType *)malloc(sizeof(ScalarType) * N);
        ScalarType *y = (ScalarType *)malloc(sizeof(ScalarType) * N);
        ScalarType *dot = (ScalarType *)malloc(sizeof(ScalarType));
        *dot = 0;

        for (int i = 0; i < N; i++) 
        {
            x[i] = 1;
            y[i] = 2;
        }

        ScalarType *cuda_x;
        ScalarType *cuda_y;
        ScalarType *cuda_dot;
        cudaMalloc(&cuda_x, sizeof(ScalarType) * N);
        cudaMalloc(&cuda_y, sizeof(ScalarType) * N);

        cudaMemcpy(cuda_x, x, sizeof(ScalarType) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_y, y, sizeof(ScalarType) * N, cudaMemcpyHostToDevice);
        timer.reset();
        for(int i=0; i < anz; ++i) 
        {
            cudaMalloc(&cuda_dot, sizeof(ScalarType));
            cudaMemcpy(cuda_dot, dot, sizeof(ScalarType), cudaMemcpyHostToDevice);

            dot_product<<<256, 256>>>(cuda_x, cuda_y, cuda_dot, N);
            cudaMemcpy(dot, cuda_dot, sizeof(ScalarType), cudaMemcpyDeviceToHost);  
            cudaDeviceSynchronize();   
        }
        myCUDA_Time = timer.get()/anz;
        std::cout << "Dot Product = " << *dot << std::endl;
        std::string sep = ";";
        csv_times << N << sep << myCUDA_Time << std::endl;
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    

    return EXIT_SUCCESS;
}