#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>

#define BLOCK_SIZE 256
#define GRID_SIZE 256

// y = A * x

__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
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

// Y= A * X
__global__ void A_MatMul_Xrm(int N, int K,
    int *csr_rowoffsets, int *csr_colindices, double *csr_values,
    double *X, double *Y)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  
    if (tid < N){
      int row_start = csr_rowoffsets[tid];
      int row_end = csr_rowoffsets[tid + 1];
  
      for (int k = 0; k < K; ++k){
        double sum = 0.0;
        for (int i = row_start; i < row_end; i++) {
          sum += csr_values[i]* X[csr_colindices[i] + k*N];
        }
        Y[k + tid*K] = sum;
      }
    }
  }

  // Y= A * X
__global__ void A_MatMul_Xcm(int N, int K,
    int *csr_rowoffsets, int *csr_colindices, double *csr_values,
    double *X, double *Y)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  
    if (tid < N){
      int row_start = csr_rowoffsets[tid];
      int row_end = csr_rowoffsets[tid + 1];
      for (int i = row_start; i < row_end; i++) {
        double aij = csr_values[i];
        int row_of_X = csr_colindices[i]*K;
        for (int k = 0; k < K; ++k){
          Y[k + tid*K] += aij * X[row_of_X + k];
        }
      }
    }
  }

__global__ void count_nonzero_entries(int* row_offsets, int N, int M) 
{
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N * M; row += gridDim.x * blockDim.x) 
    {
        int nnz_for_this_node = 1;
        int i = row / N;
        int j = row % N;

        if(i > 0) nnz_for_this_node += 1;
        if(j > 0) nnz_for_this_node += 1;
        if(i < N-1) nnz_for_this_node += 1;
        if(j < M-1) nnz_for_this_node += 1;
        
        row_offsets[row] = nnz_for_this_node;
    }
}

__global__ void assemble_A_GPU(double* values, int* columns, int* row_offsets, int N, int M) 
{
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) 
    {
        int i = row / N;
        int j = row % N;
        int counter = 0;

        if ( i > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i-1)*N+j;
            counter++;
        }
        
        if ( j > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j-1);
            counter++;
        }

        values[(int)row_offsets[row] + counter] = 4;
        columns[(int)row_offsets[row] + counter] = i*N+j;

        counter++;

        if ( j < M-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j+1);
            counter++;
        }
        if ( i < N-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i+1)*N+j;
            counter++;
        }
    }
}

/*
__global__ void assemble_A_GPU(double* values, int* columns, int* row_offsets, int N, int M) 
{
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) 
    {
        int i = row / N;
        int j = row % N;
        int counter = 0;

        if ( i > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i-1)*N+j;
            counter++;
        }
        
        if ( j > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j-1);
            counter++;
        }

        values[(int)row_offsets[row] + counter] = 4;
        columns[(int)row_offsets[row] + counter] = i*N+j;

        counter++;

        if ( j < M-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j+1);
            counter++;
        }
        if ( i < N-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i+1)*N+j;
            counter++;
        }
    }
}
*/

int main() 
{
    //std::vector<int> vec_Ns{10, 100, 300, 600, 1000, 3000, 6000, 10000, 30000, 60000, 100000, 300000, 600000, 1000000};
    std::vector<int> vec_Ns{10};
    int K = 4;
    int anz = 100;
    Timer timer;


//////////////////////////////////////////////////////////////////////////////////  
    std::fstream csv_times;
    //std::string csv_times_name = "Sparse_M_Times_Dense K=" + std::to_string(K) + " " + std::to_string(10000000) + ".csv";
    std::string csv_times_name = "Sparse_M_Times_Dense K=" + std::to_string(K) + ".csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;Time_Column_Major;Time_Row_Major;Time_Standart_K_times";
        // to csv file
    csv_times << header << std::endl;
//////////////////////////////////////////////////////////////////////////////////
    for (int& N : vec_Ns) 
    {

        std::cout << "N = " << N << std::endl;
        std::cout << "K = " << K << std::endl;

        std::cout << "Allocating host + device arrays..." << std::endl;

        double* X = (double *)malloc(sizeof(double) * N * K);
        double* Y = (double *)malloc(sizeof(double) * N * K);
        double* Y2 = (double *)malloc(sizeof(double) * N * K);
        double* y = (double *)malloc(sizeof(double) * N);

        std::fill(X, X + (N*K), 1.);
        std::fill(Y, Y + (N*K), 1.);
        std::fill(Y2, Y2 + (N*K), 1.);

        std::cout << "Fill host + device arrays..." << std::endl;

        double *cuda_X;
        double *cuda_Y;
        double *cuda_y;
        int* cuda_csr_rowoffsets; 
        int* cuda_csr_colindices;
        double* cuda_csr_values;

        cudaMalloc(&cuda_X, sizeof(double) * N*K);
        cudaMalloc(&cuda_Y, sizeof(double) * N*K);
        cudaMalloc(&cuda_y, sizeof(double) * N);

        int* csr_rowoffsets = (int* )malloc(sizeof(int) * (N+1));
        int* csr_colindices = (int* )malloc(sizeof(int) * 5*N);
        double* csr_values = (double* )malloc(sizeof(double) * 5*N);
        cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N+1));
        cudaMalloc(&cuda_csr_colindices, sizeof(int) * 5*N);
        cudaMalloc(&cuda_csr_values, sizeof(double) * 5*N);
        //
        // Copy data to GPU
        //

        std::cout << "Copying data to GPU..." << std::endl;
        cudaMemcpy(cuda_X, X, sizeof(double) * N*K, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Y, Y, sizeof(double) * N*K, cudaMemcpyHostToDevice);

        generate_fdm_laplace(sqrt(N), csr_rowoffsets, csr_colindices, csr_values);

        cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N+1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * 5*N, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5*N, cudaMemcpyHostToDevice);


        std::cout << "Running per vector product kernel K times..." << std::endl;
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            for (int k = 0; k < K; ++k)
            {
              cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, cuda_X, cuda_y);
              cudaMemcpy(y, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);
              cudaDeviceSynchronize();
            }
            
        }
        double time_single = timer.get()/anz;

        std::cout << "Running RowMajor stacked kernel..." << std::endl;
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            A_MatMul_Xrm<<<GRID_SIZE, BLOCK_SIZE>>>(
                N, K,
                cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values,
                cuda_X, cuda_Y);
            cudaMemcpy(Y, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        double time_rm_stacked = timer.get()/anz;

        std::cout << "Running ColumnMajor stacked kernel..." << std::endl;
        cudaMemcpy(cuda_Y, Y2, sizeof(double) * N*K, cudaMemcpyHostToDevice);
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            A_MatMul_Xcm<<<GRID_SIZE, BLOCK_SIZE>>>(
                N, K,
                cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values,
                cuda_X, cuda_Y);
            cudaMemcpy(Y2, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        double time_cm_stacked = timer.get()/anz;


        std::string sep = ";";
        csv_times << N << sep 
        << time_cm_stacked << sep 
        << time_rm_stacked << sep 
        << time_single
        << std::endl;

        free(X);
        free(Y);
    
        free(csr_rowoffsets); 
        free(csr_colindices);
        free(csr_values);

        cudaFree(cuda_X);
        cudaFree(cuda_Y);

        cudaFree(cuda_csr_rowoffsets); 
        cudaFree(cuda_csr_colindices);
        cudaFree(cuda_csr_values);
    }
    csv_times.close();
  std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_times_name << std::endl;
  std::cout << "Finish" << std::endl;  
    return EXIT_SUCCESS;
}