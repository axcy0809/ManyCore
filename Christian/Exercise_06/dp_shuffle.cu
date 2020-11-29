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
        int val = x[tid];
        
        sum         += val;
        maxSum      += std::abs(val);
        sqrSum      += val*val;
        maxMod      = std::abs(val) > maxMod ? val : maxMod;
        min         = val < min ? val : min;
        max         = val > max ? val :max;
        zeros       += val == 0 ? 1 : 0;
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

    int N = 100000;

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

    shuffleKernel<<<N/256, 128>>>(cuda_x, cuda_y, N);

    cudaMemcpy(y, cuda_y, sizeof(int) * 7, cudaMemcpyDeviceToHost);

    //std::cout << "Input" << std::endl;
    //printContainer(x, N);
    //std::cout << std::endl;

    std::cout << "Sum of all entries: " << y[0] << std::endl;
    std::cout << "Sum of maximum values: " << y[1] << std::endl;
    std::cout << "Sum of squares: " << y[2] << std::endl;
    std::cout << "Max-norm: " << y[3] << std::endl;
    std::cout << "minimum value: " << y[4] << std::endl;
    std::cout << "maximum value: " << y[5] << std::endl;
    std::cout << "number of zeros: " << y[6] << std::endl; 

    return EXIT_SUCCESS;
}