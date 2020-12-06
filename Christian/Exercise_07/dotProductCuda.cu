#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "timer.hpp"
#include <random>


__global__ void dot_product(double *x, double *y, double *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ double cache[256];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2) 
    {
        __syncthreads();
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
    }

	if(threadIdx.x == 0){
		atomicAdd(dot, cache[0]);
	}
}



int main()
{
	unsigned int n = 128*1024;
	double *h_prod;
	double *d_prod;
	double *h_x, *h_y;
	double *d_x, *d_y;
    Timer timer;

    h_prod = new double[n];
    h_x = new double[n];
    h_y = new double[n];


    // fill host array with data
	for(unsigned int i=0;i<n;i++){
		h_x[i] = 3;
		h_y[i] = 2;
	}

    // start timer
    std::vector<double> timings;
    for(int reps=0; reps < 10; ++reps) {
        timer.reset();
	
		// allocate memory
		cudaMalloc(&d_prod, sizeof(double));
		cudaMalloc(&d_x, n*sizeof(double));
		cudaMalloc(&d_y, n*sizeof(double));
		cudaMemset(d_prod, 0.0, sizeof(double));
	
	
		// copy data to device
		cudaMemcpy(d_x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, h_y, n*sizeof(double), cudaMemcpyHostToDevice);
	
	
		dot_product<<<256, 256>>>(d_x, d_y, d_prod, n);
	
		// copy data back to host
		cudaMemcpy(h_prod, d_prod, sizeof(double), cudaMemcpyDeviceToHost);
	
    	// get runtime
    	timings.push_back(timer.get());
    }

    std::sort(timings.begin(), timings.end());
    double time_elapsed = timings[10/2];
	

	// report results
	std::cout<<"dot product computed on GPU is: "<<*h_prod<<" and took " << time_elapsed << " s" <<std::endl;


	// free memory
	free(h_prod);
	free(h_x);
	free(h_y);
	cudaFree(d_prod);
	cudaFree(d_x);
	cudaFree(d_y);

}