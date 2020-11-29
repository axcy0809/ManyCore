#include <iostream>

__global__ void sharedMemoryKernel(const  int* x,  int* y, const  int N) {
    __shared__  int sharedMemory[7][256];
     int sum = 0;
     int maxSum = 0;
     int sqrSum = 0;
     int maxMod = 0;
     int min = x[0];
     int max = 0;
     int zeros = 0;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < N; tid += gridDim.x * blockDim.x) {
        int val = x[tid];
        
        sum += val;
        maxSum += std::abs(val);
        sqrSum += val*val;
        maxMod = std::abs(val) > maxMod ? val : maxMod;
        min = val < min ? val : min;
        max = val > max ? val :max;
        zeros += val == 0 ? 1 : 0;
    }

    int tid = threadIdx.x;
    if (tid < N) {
        sharedMemory[0][threadIdx.x] = sum;
        sharedMemory[1][threadIdx.x] = maxSum;
        sharedMemory[2][threadIdx.x] = sqrSum;
        sharedMemory[3][threadIdx.x] = maxMod;
        sharedMemory[4][threadIdx.x] = min;
        sharedMemory[5][threadIdx.x] = max;
        sharedMemory[6][threadIdx.x] = zeros;

        __syncthreads();
        // blockDim.x needs to be a power of 2 in order for this to work
        for (int i = blockDim.x/2; i != 0; i /= 2) {
            __syncthreads();
            if (tid < i) {
                sharedMemory[0][tid] += sharedMemory[0][tid + i];
                sharedMemory[1][tid] += sharedMemory[1][tid + i];
                sharedMemory[2][tid] += sharedMemory[2][tid + i];
                sharedMemory[3][tid] = sharedMemory[3][tid] > sharedMemory[3][tid + i] ? sharedMemory[3][tid] : sharedMemory[3][tid + i];
                sharedMemory[4][tid] = sharedMemory[4][tid] < sharedMemory[4][tid + i] ? sharedMemory[4][tid] : sharedMemory[4][tid + i];
                sharedMemory[5][tid] = sharedMemory[5][tid] > sharedMemory[5][tid + i] ? sharedMemory[5][tid] : sharedMemory[5][tid + i];
                sharedMemory[6][tid] += sharedMemory[6][tid + i];
            }
        }
    }

    if (tid == 0) {
        atomicAdd(y, sharedMemory[0][0]);
        atomicAdd(y+1, sharedMemory[1][0]);
        atomicAdd(y+2, sharedMemory[2][0]);
        atomicMax(y+3, sharedMemory[3][0]);
        atomicMin(y+4, sharedMemory[4][0]);
        atomicMax(y+5, sharedMemory[5][0]);
        atomicAdd(y+6, sharedMemory[6][0]);
    }
}

template <typename T>
 void printContainer(T container,  int N) {
     for (int i = 0; i < N; i++) {
         std::cout << container[i] << " | ";
    }
}


int main() {

    int N = 5;

     int *x = ( int *)malloc(sizeof( int) * N);
     int *y = ( int *)malloc(sizeof( int) * 7);

    for (int i = 0; i < N; i++) {
        x[i] = i - N/2;
    }

     int *cuda_x;
     int *cuda_y;
    cudaMalloc(&cuda_x, sizeof( int) * N);
    cudaMalloc(&cuda_y, sizeof( int) * 7);

    cudaMemcpy(cuda_x, x, sizeof( int) * N, cudaMemcpyHostToDevice);

    sharedMemoryKernel<<<256, 256>>>(cuda_x, cuda_y, N);

    cudaMemcpy(y, cuda_y, sizeof( int) * 7, cudaMemcpyDeviceToHost);

    std::cout << "Input" << std::endl;
    printContainer(x, N);
    std::cout << std::endl;

    std::cout << "Sum of all entries: " << y[0] << std::endl;
    std::cout << "Sum of maximum values: " << y[1] << std::endl;
    std::cout << "Sum of squares: " << y[2] << std::endl;
    std::cout << "Max-norm: " << y[3] << std::endl;
    std::cout << "minimum value: " << y[4] << std::endl;
    std::cout << "maximum value: " << y[5] << std::endl;
    std::cout << "number of zeros: " << y[6] << std::endl; 

    free(x);
    free(y);
    cudaFree(cuda_x);
    cudaFree(cuda_y);

    return EXIT_SUCCESS;
}