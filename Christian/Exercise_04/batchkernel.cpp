__global__ void cuda_many_dot_product(int N, double *x, double **y, double *result)
{
  __shared__ double shared_mem_0[512];
  __shared__ double shared_mem_1[512];
  __shared__ double shared_mem_2[512];
  __shared__ double shared_mem_3[512];
  __shared__ double shared_mem_4[512];
  __shared__ double shared_mem_5[512];
  __shared__ double shared_mem_6[512];
  __shared__ double shared_mem_7[512];
 
  double dot_0 = 0;
  double dot_1 = 0;
  double dot_2 = 0;
  double dot_3 = 0;
  double dot_4 = 0;
  double dot_5 = 0;
  double dot_6 = 0;
  double dot_7 = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double val = x[i];
    dot_0 += val * y[0][i];
    dot_1 += val * y[1][i];
    dot_2 += val * y[2][i];
    dot_3 += val * y[3][i];
    dot_4 += val * y[4][i];
    dot_5 += val * y[5][i];
    dot_6 += val * y[6][i];
    dot_7 += val * y[7][i];
  }
 
  shared_mem_0[threadIdx.x] = dot_0;
  shared_mem_1[threadIdx.x] = dot_1;
  shared_mem_2[threadIdx.x] = dot_2;
  shared_mem_3[threadIdx.x] = dot_3;
  shared_mem_4[threadIdx.x] = dot_4;
  shared_mem_5[threadIdx.x] = dot_5;
  shared_mem_6[threadIdx.x] = dot_6;
  shared_mem_7[threadIdx.x] = dot_7;

  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem_0[threadIdx.x] += shared_mem_0[threadIdx.x + k];
      shared_mem_1[threadIdx.x] += shared_mem_1[threadIdx.x + k];
      shared_mem_2[threadIdx.x] += shared_mem_2[threadIdx.x + k];
      shared_mem_3[threadIdx.x] += shared_mem_3[threadIdx.x + k];
      shared_mem_4[threadIdx.x] += shared_mem_4[threadIdx.x + k];
      shared_mem_5[threadIdx.x] += shared_mem_5[threadIdx.x + k];
      shared_mem_6[threadIdx.x] += shared_mem_6[threadIdx.x + k];
      shared_mem_7[threadIdx.x] += shared_mem_7[threadIdx.x + k];
    }
  }
 
  if (threadIdx.x == 0){
       atomicAdd(result+0, shared_mem_0[0]);
       atomicAdd(result+1, shared_mem_1[0]);
       atomicAdd(result+2, shared_mem_2[0]);
       atomicAdd(result+3, shared_mem_3[0]);
       atomicAdd(result+4, shared_mem_4[0]);
       atomicAdd(result+5, shared_mem_5[0]);
       atomicAdd(result+6, shared_mem_6[0]);
       atomicAdd(result+7, shared_mem_7[0]);
  }
}
 