# include <stdio.h>
# include "timer.hpp"
# include <iostream>
# include <numeric>
# include <vector>
#include <fstream>
#include <string>
#include "cublas_v2.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 128

// ------------------ KERNELS ---------------

/** atomicMax for double
 * 
 * References:
 * (1) https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicmax
 * (2) https://www.micc.unifi.it/bertini/download/gpu-programming-basics/2017/gpu_cuda_5.pdf
 * (3) https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
 __device__ void atomicMax(double* address, double val){    
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
      // atomicCAS returns the value that is stored in address AFTER the CAS
      // atomicCAS(a, b, c) --> return c
      //
    } while (assumed != old);
  }
  
  /** atomicMin for double
   */
  __device__ void atomicMin(double* address, double val){    
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
      // atomicCAS returns the value that is stored in address AFTER the CAS
      // atomicCAS(a, b, c) --> return c
      //
    } while (assumed != old);
  }

__global__ void dot_pro(double *x, double *tmp, int N)
{
    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    __shared__ double cache[BLOCK_SIZE];

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
    __shared__ double sumofallelements[BLOCK_SIZE];
    __shared__ double einsnorm[BLOCK_SIZE];
    __shared__ double zweisnorm[BLOCK_SIZE];
    __shared__ double maxnorm[BLOCK_SIZE];
    __shared__ double maxval[BLOCK_SIZE];
    __shared__ double minval[BLOCK_SIZE];
    __shared__ double zeros[BLOCK_SIZE];

    if (blockDim.x * blockIdx.x < N)
    {
        unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
        unsigned int str = blockDim.x*gridDim.x;
        double sum = 0;
        double einssum = 0;
        double zweissum = 0;
        double max = x[0];
        double min = max;
        double count = 0;
        while(ind < N)
        {
            for (int i = 0; i < N; i += str) 
            {
                sum += x[i]; // sum of all entries
                einssum += std::abs(x[i]);  // 1-norm
                zweissum += x[i] * x[i];  // 2-norm
                max = fmax(x[i], max);  // max_value
                min = fmin(x[i], min);  // min_value
                if (x[i] == 0)  // counts the zero entries
                {
                    count = count + 1;
                }
            }
            ind += str;
        }
        sumofallelements[threadIdx.x] = sum;
        einsnorm[threadIdx.x] = einssum;
        zweisnorm[threadIdx.x] = zweissum;
        maxnorm[threadIdx.x] = fmax(std::abs(min), max);
        maxval[threadIdx.x] = max;
        minval[threadIdx.x] = min;
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
                maxnorm[threadIdx.x] = fmax(maxnorm[threadIdx.x + i], maxnorm[threadIdx.x]);
                minval[threadIdx.x] = fmin(minval[threadIdx.x + i], minval[threadIdx.x]); 
                maxval[threadIdx.x] = fmax(maxval[threadIdx.x + i], maxval[threadIdx.x]); 
                zeros[threadIdx.x] += zeros[threadIdx.x + i];
            }
        }
        if(threadIdx.x == 0)
        {
            atomicAdd(dot + 0,sumofallelements[0]);
            atomicAdd(dot + 1,einsnorm[0]);
            atomicAdd(dot + 2,std::sqrt(zweisnorm[0]));
            atomicMax(dot + 3, maxnorm[0]);
            atomicMin(dot + 4, minval[0]);
            atomicMax(dot + 5, maxval[0]);
            atomicAdd(dot + 6, zeros[0]);
        }
    }
}

__global__ void analyze_x_warp(double *x, double *results, int N) 
{
    if (blockDim.x * blockIdx.x < N) {
      int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
      const int stride = blockDim.x * gridDim.x;
  
      double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
      // double mod_max = 0.0;
      double max = x[0];
      double min = max;
      int z_entries = 0;
      for (; tid < N; tid += stride) {
        double value = x[tid];
        sum += value;
        abs_sum += std::abs(value);
        sqr_sum += value*value;
  
        min = fmin(value, min); 
        max = fmax(value, max);
        z_entries += (value)? 0 : 1;
      }
      tid = threadIdx.x; // block tid 
      double mod_max = fmax(std::abs(min), max);
  
      __syncthreads();
      for (int i = warpSize / 2; i != 0; i /= 2) {
        //__syncthreads();
        sum += __shfl_down_sync(0xffffffff, sum, i);
        abs_sum += __shfl_down_sync(0xffffffff, abs_sum, i);
        sqr_sum += __shfl_down_sync(0xffffffff, sqr_sum, i);
  
        double tmp = __shfl_down_sync(0xffffffff, mod_max, i);
        mod_max = fmax(tmp, mod_max);
        tmp = __shfl_down_sync(0xffffffff, min, i);
        min = fmin(tmp, min);
        tmp = __shfl_down_sync(0xffffffff, max, i);
        max = fmax(tmp, max) ;
  
        z_entries += __shfl_down_sync(0xffffffff, z_entries, i);
      }  
      if (tid % warpSize == 0) // a block can consist of multiple warps
      {
        atomicAdd(results, sum);
        atomicAdd(results+1, abs_sum);
        atomicAdd(results+2, sqr_sum);
  
        atomicMax(results+3, mod_max);
        atomicMin(results+4, min);
        atomicMax(results+5, max);
  
        atomicAdd(results+6, z_entries);
      }
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

__global__ void analyze_x_shared(double *x, double *results,const int N) 
{
    if (blockDim.x * blockIdx.x < N) {
      int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
      const int stride = blockDim.x * gridDim.x;
  
      __shared__ double cache[7][BLOCK_SIZE];
  
      double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
      // double mod_max = 0.0;
      double max = x[0];
      double min = max;
      double z_entries = 0;
      for (; tid < N; tid += stride) {
        double value = x[tid];
        sum += value;
        abs_sum += std::abs(value);
        sqr_sum += value*value;
  
        // mod_max = (std::abs(value) > mod_max)? value : mod_max;
        min = fmin(value, min); 
        max = fmax(value, max);
        z_entries += (value)? 0 : 1;
      }
      tid = threadIdx.x; // block tid 
      cache[0][tid] = sum;
      cache[1][tid] = abs_sum;
      cache[2][tid] = sqr_sum;
      cache[3][tid] = fmax(std::abs(min), max);
      cache[4][tid] = min;
      cache[5][tid] = max;
      cache[6][tid] = z_entries;
  
      __syncthreads();
      for (int i = blockDim.x / 2; i != 0; i /= 2) {
        __syncthreads();
        if (tid < i) { // lower half does smth, rest idles
          // sums
          cache[0][tid] += cache[0][tid + i]; 
          cache[1][tid] += cache[1][tid + i]; 
          cache[2][tid] += cache[2][tid + i]; 
          // min/max values
          cache[3][tid] = fmax(cache[3][tid + i], cache[3][tid]); // already all values are std::abs(...)
          cache[4][tid] = fmin(cache[4][tid + i], cache[4][tid]); 
          cache[5][tid] = fmax(cache[5][tid + i], cache[5][tid]); 
  
          // "sum"
          cache[6][tid] += cache[6][tid + i]; 
        }
      }
  
      if (tid == 0) // cache[0] now contains block_sum
      {
        atomicAdd(results, cache[0][0]);
        atomicAdd(results+1, cache[1][0]);
        atomicAdd(results+2, std::sqrt(cache[2][0]));
  
        // Ideally...
        atomicMax(results+3, cache[3][0]);
        atomicMin(results+4, cache[4][0]);
        atomicMax(results+5, cache[5][0]);
  
        atomicAdd(results+6, cache[6][0]);
      }
    }
}
/*
template <typname T>
    void printContainer(T container, int N)
    {
        for (int i = 0; i < N; i++)
        {
            std::cout << container[i] <<  " | ";
        }
    }
*/




int main(void)
{
  std::vector<int> vec_Ns{10, 100, 1000, 10000, 30000, 100000,1000000, 10000000, 100000000};
  //std::vector<int> vec_Ns{6};
  /*
  std::vector<int> vec_Ns;
  for (int i = 10; i < 100000; i = i + 2000)
  {
    vec_Ns.push_back(i);
  }
  */
  double *cuda_x, *cuda_cublas_dot, *cuda_cublas_sum;
  int *cuda_cublas_max, *cuda_cublas_min;
  double *x;
  double dot_cublas=0.;
  double *results, *cuda_result;
  double *wshuffel, *cuda_wshuffel;
  double *dot, *cuda_dot;

  Timer timer;
  int anz = 100;
  std::vector<double> ownruntime;
  int shift = 0;
//////////////////////////////////////////////////////////////////////////////////  
  std::fstream csv_times;
  std::string csv_times_name = "shuffeld_adopt.csv";
  csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
  std::string header = "N;Time_CUBLAS_DotProduct;Time_DotProd_shared;Time_mmsszz_shared;Time_mmsszz_worp_shuffel;Adapt;Time_mmsszz_worp_shuffel_addapt";//Time_CUBLAS_mmsszz";
      // to csv file
  csv_times << header << std::endl;
//////////////////////////////////////////////////////////////////////////////////
  for (int& N : vec_Ns) 
  {  
    double refsum = 0, ref1norm = 0, ref2norm = 0, refmaxnorm = 0, refmin = 0, refmax = 0, refzeros = 0;
    int adapt_gridsize = N / BLOCK_SIZE;
    //
    // generates a random object:
    //
    srand (time(NULL));
    std::cout << "N: " << N << std::endl;
    //
    // init CUBLAS
    //
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    //
    // init arrays
    //
    x = (double*)malloc(sizeof(double) * N);
    results = (double*)malloc(sizeof(double) * 7);
    dot = (double*)malloc(sizeof(double));
    wshuffel = (double*)malloc(sizeof(double));

    for(size_t i=0; i<N; ++i) 
    {
        //x[i] = rand() % 10 + 1;
        x[i] = 1;
    }
    /*

    x[0] = 1;
    x[1] = 1;
    x[2] = 0;
    x[3] = 2;
    x[4] = 3;
    x[5] = -4;

    for(size_t i=0; i<N; ++i) 
    {
        std::cout << x[i] << " | ";
    }
    std::cout << " " << std::endl;
    */



/////////////////////////////// reference value ///////////////////////////////////////////////////////// 

    refmax = x[0];
    refmin = refmax;
    refzeros = 0;

    for(size_t i=0; i<N; ++i)   
    {
        refsum += x[i];
        ref1norm += std::abs(x[i]);
        ref2norm += x[i]*x[i];
        refmin = fmin(x[i], refmin); 
        refmax = fmax(x[i], refmax);
        if (x[i] == 0)  // counts the zero entries
        {
            refzeros = refzeros + 1;
        }
    }
    ref2norm = std::sqrt(ref2norm);
    refmaxnorm = fmax(std::abs(refmin), refmax);
    
    cudaMalloc(&cuda_x, N*sizeof(double));
    cudaMalloc( (void **)(&cuda_cublas_dot), sizeof(double) );
    cudaMalloc( (void **)(&cuda_cublas_sum), sizeof(double) );
    cudaMalloc( (void **)(&cuda_cublas_max), sizeof(int) );
    cudaMalloc( (void **)(&cuda_cublas_min), sizeof(int) );
    cudaMalloc((void **)(&cuda_result), 7*sizeof(double));
    cudaMalloc((void **)(&cuda_wshuffel), 7*sizeof(double));
    cudaMalloc(&cuda_dot, sizeof(double) );




    cudaMemcpy(cuda_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

//////////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        cublasDdot(h, N, cuda_x, 1, cuda_x, 1, cuda_cublas_dot);
    }
    cudaMemcpy(&dot_cublas, cuda_cublas_dot, sizeof(double), cudaMemcpyDeviceToHost);
    ownruntime.push_back (1000*timer.get()/anz);
    

//////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        dot_pro<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_x, cuda_dot, N);
    }
    //cudaDeviceSynchronize();

    cudaMemcpy(dot,cuda_dot, sizeof(double), cudaMemcpyDeviceToHost);
    ownruntime.push_back (1000*timer.get()/anz);
    //////////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
      mmsszz<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_x, cuda_result, N);
    }
    //cudaDeviceSynchronize();
    
    cudaMemcpy(results,cuda_result, 7*sizeof(double), cudaMemcpyDeviceToHost);
    ownruntime.push_back (1000*timer.get()/anz);

//////////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        analyze_x_warp<<<adapt_gridsize, BLOCK_SIZE>>>(cuda_x, cuda_wshuffel, N);
    }
    cudaMemcpy(wshuffel, cuda_wshuffel, sizeof(double), cudaMemcpyDeviceToHost);
    ownruntime.push_back (1000*timer.get()/anz);

//////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        analyze_x_warp<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_x, cuda_wshuffel, N);
    }
    cudaMemcpy(wshuffel, cuda_wshuffel, sizeof(double), cudaMemcpyDeviceToHost);
    ownruntime.push_back (1000*timer.get()/anz);

//////////////////////////////////////////////////////////////////////////////////
/*
    cudaDeviceSynchronize();
    timer.reset();
    for (int i = 0; i < anz; i++)
    {
        cublasDasum(h, N, cuda_x, sizeof(double), cuda_cublas_sum);
        cublasIdamax(h,N, cuda_x,sizeof(double), cuda_cublas_max);
        cublasIdamin(h,N, cuda_x,sizeof(double), cuda_cublas_min);
        cublasDnrm2(h, N, cuda_x, sizeof(double), cuda_cublas_sum);
    }
    cudaDeviceSynchronize();

    ownruntime.push_back (1000*timer.get()/anz);
*/

    

/*
    std::cout << "Sum of all elements: " << results[0] << " Ref of Sum all elements: " << refsum << std::endl;
    std::cout << "1Norm: " << results[1] << " Ref of 1Norm: " << ref1norm << std::endl;
    std::cout << "2Norm: " << results[2] << " Ref of 2Norm: " << ref2norm << std::endl;
    std::cout << "MaxNorm: " << results[3] << " Ref of MaxNorm: " << refmaxnorm << std::endl;
    std::cout << "minvalue: " << results[4] << " Ref of minxalue: " << refmin << std::endl;
    std::cout << "maxvalue: " << results[5] << " Ref of maxvalue: " << refmax << std::endl;
    std::cout << "nomOfzeros: " << results[6] << " Ref of nomOfzeros: " << refzeros << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
*/
    std::cout << "Time_CUBLAS DotProduct: " << ownruntime[shift] << std::endl;
    std::cout << "Time_DotProd_shared: " << ownruntime[shift + 1] << std::endl;
    std::cout << "Time_mmsszz_shared: " << ownruntime[shift + 2] << std::endl;
    std::cout << "Time_mmsszz_worp_shuffel: " << ownruntime[shift + 3] << std::endl;
    std::cout << "Time_mmsszz_worp_shuffel_addapt: "<< ownruntime[shift + 4] << " Adapt: " << adapt_gridsize << ownruntime[shift + 4] << std::endl;
    //std::cout << "Time_CUBLAS_mmsszz: " << ownruntime[shift + 5] << std::endl;

    std::string sep = ";";
    csv_times << N << sep 
    << ownruntime[shift] << sep 
    << ownruntime[shift + 1] << sep 
    << ownruntime[shift + 2] << sep 
    << ownruntime[shift + 3] << sep
    << adapt_gridsize << sep
    << ownruntime[shift + 4]
    //<< ownruntime[shift + 5] 
    << std::endl;


    cublasDestroy(h);
    cudaFree(cuda_result);
    cudaFree(cuda_x);
    cudaFree(cuda_cublas_sum);
    cudaFree(cuda_cublas_dot);

    cudaFree(cuda_cublas_max);
    cudaFree(cuda_cublas_max);
    cudaFree(cuda_wshuffel);
    cudaFree(cuda_dot);
  
    free(x);
    free(results);
    free(dot);
    free(wshuffel);
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

