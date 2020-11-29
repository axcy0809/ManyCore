#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <iostream>

__global__ void shuffleKernel(const int* x, int* y, const int N) {

    int sum             = 0;
    int maxSum          = 0;
    int sqrSum          = 0;
    int maxMod          = 0;
    int min             = x[0];
    int max             = 0;
    int zeros           = 0;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < N; tid += gridDim.x * blockDim.x) {
        if (tid < N) { // this if is actually important for when N is smaller than gridsize*blocksize
            int val = x[tid];

            sum         += val;
            maxSum      += std::abs(val);
            sqrSum      += val*val;
            maxMod      = std::abs(val) > maxMod ? val : maxMod;
            min         = val < min ? val : min;
            max         = val > max ? val :max;
            zeros       += val == 0 ? 1 : 0;
        }
    }

    int tid = threadIdx.x;
    for (int i = warpSize / 2; i != 0; i /= 2) {
        sum         += __shfl_down_sync(0xffffffff, sum, i);
        maxSum      += __shfl_down_sync(0xffffffff, maxSum, i);
        sqrSum      += __shfl_down_sync(0xffffffff, sqrSum, i);
        int temporary = __shfl_down_sync(0xffffffff, maxMod, i);
        maxMod      = temporary > maxMod ? temporary : maxMod;
        temporary   = __shfl_down_sync(0xffffffff, min, i);
        min         = temporary < min ? temporary : min;
        temporary   = __shfl_down_sync(0xffffffff, max, i);
        max         = temporary > max ? temporary : max;
        zeros       += __shfl_down_sync(0xffffffff, zeros, i);
    }
    __syncthreads();
    if (tid % warpSize == 0) {
        atomicAdd(y,   sum);
        atomicAdd(y+1, maxSum);
        atomicAdd(y+2, sqrSum);
        atomicMax(y+3, maxMod);
        atomicMin(y+4, min);
        atomicMax(y+5, max);
        atomicAdd(y+6, zeros);
    }
}

template <typename T>
 void printContainer(T container, int N) {
     for (int i = 0; i < N; i++) {
         std::cout << container[i] << " | ";
    }
}


int main() {

    Timer timer;
    int N = 1000;

    int *x = (int *)malloc(sizeof(int) * N);
    int *y = (int *)malloc(sizeof(int) * 7);

    for (int i = 0; i < N; i++) {
        x[i] = i - N/2;
    }

    int *cuda_x;
    int *cuda_y;
    cudaMalloc(&cuda_x, sizeof(int) * N);
    cudaMalloc(&cuda_y, sizeof(int) * 7);

    cudaMemcpy(cuda_x, x, sizeof(int) * N, cudaMemcpyHostToDevice);

    std::vector<double> timings;
    for(int reps=0; reps < 10; ++reps) {
        timer.reset();
        shuffleKernel<<<N/256, 128>>>(cuda_x, cuda_y, N);
        cudaMemcpy(y, cuda_y, sizeof(int) * 7, cudaMemcpyDeviceToHost);
        timings.push_back(timer.get());
    }

    std::sort(timings.begin(), timings.end());
    double time_elapsed = timings[10/2];

    std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;

    return EXIT_SUCCESS;
}