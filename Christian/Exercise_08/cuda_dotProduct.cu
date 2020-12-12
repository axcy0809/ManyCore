#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iostream>

__global__ void dot_product(int* x, int* y, int* dot, int N) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ int cache[128];

    int temp = 0;
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


int main() {

    Timer timer;
    int N = 1000;

    int *x = (int *)malloc(sizeof(int) * N);
    int *y = (int *)malloc(sizeof(int) * N);
    int *dot = (int *)malloc(sizeof(int));

    for (int i = 0; i < N; i++) {
        x[i] = 1;
        y[i] = 2;
    }
    *dot = 0;

    int *cuda_x;
    int *cuda_y;
    int *cuda_dot;
    cudaMalloc(&cuda_x, sizeof(int) * N);
    cudaMalloc(&cuda_y, sizeof(int) * N);
    cudaMalloc(&cuda_dot, sizeof(int));

    cudaMemcpy(cuda_x, x, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y, y, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_dot, dot, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<double> timings;
    for(int reps=0; reps < 10; ++reps) {
        timer.reset();
        dot_product<<<N/256, 128>>>(cuda_x, cuda_y, cuda_dot, N);
        cudaMemcpy(dot, cuda_dot, sizeof(int), cudaMemcpyDeviceToHost);
        timings.push_back(timer.get());        
        std::cout << "Dot Product = " << *dot << std::endl;
        *dot = 0;
        cudaMemcpy(cuda_dot, dot, sizeof(int), cudaMemcpyHostToDevice);
    }

    std::sort(timings.begin(), timings.end());
    double time_elapsed = timings[10/2];

    std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;

    

    return EXIT_SUCCESS;
}