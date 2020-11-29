#include <iostream>

// y = A * x
__global__ void sparseVector(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;

    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }

    y[i] = sum;
  }
}


// Y = A * X
__global__ void sparseDenseRowMajor(int N, int K, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

    for (int k = 0; k < K; k++) {
      double sum = 0;

      for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
        sum += csr_values[jj] * x[csr_colindices[jj] + N*k];
      }

      y[i + N*k] = sum;
    }

  }
}


// Y = A * X
__global__ void sparseDenseColumnMajor(int N, int K, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

    for (int k = 0; k < K; k++) {
      double sum = 0;

      for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
        sum += csr_values[jj] * x[csr_colindices[jj] + N*k];
      }

      y[i + N*k] = sum;
    }

  }
}


template <typename T>
 void printContainer(T container,  int N) {
     for (int i = 0; i < N; i++) {
         std::cout << container[i] << " | ";
    }
    std::cout << std::endl;
}


int main() {

  int N = 5;
  int K = 3;

  double *values = ( double *)malloc(sizeof( double) * 9);
  int *colindices = (int*)malloc(sizeof(int) * 9);
  int *offsets = (int*)malloc(sizeof(int) * 6);

  double *x = ( double *)malloc(sizeof( double) * N);
  double *X = (double*)malloc(sizeof(double) * N*K);

  double *y = ( double *)malloc(sizeof( double) * N);
  double *Y = (double*)malloc(sizeof(double) * N*K);

  values[0] = 2;
  values[1] = 7;
  values[2] = 4;
  values[3] = 1;
  values[4] = 9;
  values[5] = 3;
  values[6] = 6;
  values[7] = 5;
  values[8] = 8;

  colindices[0] = 1;
  colindices[1] = 3;
  colindices[2] = 4;
  colindices[3] = 2;
  colindices[4] = 3;
  colindices[5] = 0;
  colindices[6] = 2;
  colindices[7] = 4;
  colindices[8] = 1;

  offsets[0] = 0;
  offsets[1] = 3;
  offsets[2] = 5;
  offsets[3] = 6;
  offsets[4] = 8;
  offsets[5] = 9;

  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;
  x[4] = 5;

  X[0] = 1;
  X[1] = 2;
  X[2] = 3;
  X[3] = 4;
  X[4] = 5;
  X[5] = 1;
  X[6] = 2;
  X[7] = 3;
  X[8] = 4;
  X[9] = 5;
  X[10] = 1;
  X[11] = 2;
  X[12] = 3;
  X[13] = 4;
  X[14] = 1;

  double    *cuda_values;
  int       *cuda_colindices;
  int       *cuda_offsets;
  double    *cuda_x;
  double    *cuda_X;
  double    *cuda_y;
  double    *cuda_Y;

  cudaMalloc(&cuda_values, sizeof( double) * 9);
  cudaMalloc(&cuda_colindices, sizeof( int) * 9);
  cudaMalloc(&cuda_offsets, sizeof( int) * 6);
  cudaMalloc(&cuda_x, sizeof( double) * N);
  cudaMalloc(&cuda_X, sizeof( double) * N*K);
  cudaMalloc(&cuda_y, sizeof( double) * N);
  cudaMalloc(&cuda_Y, sizeof( double) * N*K);

  cudaMemcpy(cuda_values, values, sizeof( double) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_colindices, colindices, sizeof( int) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_offsets, offsets, sizeof( int) * 6, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_x, x, sizeof( double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_X, X, sizeof( double) * N*K, cudaMemcpyHostToDevice);

  sparseVector<<<256, 256>>>(N, cuda_offsets, cuda_colindices, cuda_values, cuda_x, cuda_y);
  cudaMemcpy(y, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

  sparseDense<<<256, 256>>>(N, K, cuda_offsets, cuda_colindices, cuda_values, cuda_X, cuda_Y);
  cudaMemcpy(Y, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
  

  std::cout << "x : " << std::endl;
  printContainer(x, N);
  std::cout << "Result : " << std::endl;
  printContainer(y, N);

  std::cout << std::endl;
  std::cout << "X : " << std::endl;
  printContainer(X, N*K);
  std::cout << "Result : " << std::endl;
  printContainer(Y, N*K);

  free(x);
  free(X);
  free(y);
  free(Y);
  free(values);
  free(colindices);
  free(offsets);

  cudaFree(cuda_values);
  cudaFree(cuda_colindices);
  cudaFree(cuda_offsets);
  cudaFree(cuda_x);
  cudaFree(cuda_X);
  cudaFree(cuda_y);
  cudaFree(cuda_Y);


  return EXIT_SUCCESS;
}
