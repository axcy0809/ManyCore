//
// Tutorial for demonstrating a simple OpenCL vector addition kernel
//
// Author: Karl Rupp    rupp@iue.tuwien.ac.at
//

typedef double       ScalarType;


#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"

/*
const char *my_opencl_program = ""
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
""
"__kernel void vec_add (__global double *x,\n"
"                       __global double *y,\n"
"                      __global double *result,\n"
"                    __local double *localSums)\n"
"{\n"
"uint local_id = get_local_id(0);\n"
"uint group_size = get_local_size(0);\n"
"localSums[local_id] = x[get_global_id(0)] * y[get_global_id(0)];\n"
"for (uint stride = group_size/2; stride>0; stride /=2)\n"
"{\n"
"     barrier(CLK_LOCAL_MEM_FENCE);\n"
"     if (local_id < stride)\n"
"       localSums[local_id] += localSums[local_id + stride];"
"}\n"
"if (local_id == 0)\n"
"result[get_group_id(0)] = localSums[0];\n"
"}";  
*/
const char *my_opencl_program = ""
"__kernel void vec_add(__global double *x,\n"
"                      __global double *y,\n"
"                      __global double *result,\n"
"                      unsigned int N\n)"
"{\n"
"   unsigned int gid = get_global_id(0);\n"
"  for (unsigned int i  = gid;\n"
"                    i  < N;\n"
"                    i += gid)\n"
"    result[i] = x[i] * y[i];\n"
"}"; 


// you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.
/*
"__kernel void vec_add(__global double *x,\n"
"                      __global double *y,\n"
"                      __global double *result,\n"
"                      unsigned int N\n)"
"{\n"
"   double sum = 0;\n"
"   __local double cache[128];\n"
"   unsigned int gid = get_global_id(0);\n"
"  for (unsigned int i  = get_global_id(0);\n"
"                    i  < N;\n"
"                    i += get_global_size(0))\n"
"    sum += x[i] * y[i];\n"
"     result[0] = sum;\n"
"}"
*/

/*
  __kernel void sumGPU ( __global const double *input, 
                         __global double *partialSums,
                         __local double *localSums)
 {
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Copy from global to local memory
  localSums[local_id] = input[get_global_id(0)];

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride /=2)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
 }      

*/
/*
    for(int i = blockDim.x/2; i>0; i/=2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
    }
*/

/*"   for(int i = get_local_id(0)/2; i>0; i/=2)\n"
"{\n"
"     if(get_global_id(0) < i)\n"
"     {\n"
"         cache[get_global_id(0)] += cache[get_global_id(0) + i];\n"
"     }\n"
"}\n"
"     if(get_local_id(0))\n"
"{\n"
"     result[get_local_id(0)] = cache[0];\n"
"}\n"
*/

void CPU_dot(ScalarType *x, ScalarType *y, ScalarType *dot, unsigned int N)
{
  /*
  for (int i = 0; i < N; i++)
  {
    dot[i] = x[i] * y[i];
  }
  */
 std::cout << "Hallo" << std::endl;
}
/*
__global__ void GPU_dot(double *x, double *y, double *tmp, unsigned int N)
{
    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    while(ind < N)
    {
        tmp[i] = x[ind]*y[ind];
        ind += str;
    }
}
*/

int main()
{
  cl_int err;

  //
  /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
  //

  //
  // Query platform:
  //
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
  std::cout << "# Platforms found: " << num_platforms << std::endl;
  cl_platform_id my_platform = platform_ids[0];


  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); OPENCL_ERR_CHECK(err);
  std::cout << "# Devices found: " << num_devices << std::endl;
  cl_device_id my_device_id = device_ids[0];

  char device_name[64];
  size_t device_name_len = 0;
  err = clGetDeviceInfo(my_device_id, CL_DEVICE_NAME, sizeof(char)*63, device_name, &device_name_len); OPENCL_ERR_CHECK(err);
  std::cout << "Using the following device: " << device_name << std::endl;

  //
  // Create context:
  //
  cl_context my_context = clCreateContext(0, 1, &my_device_id, NULL, NULL, &err); OPENCL_ERR_CHECK(err);


  //
  // create a command queue for the device:
  //
  cl_command_queue my_queue = clCreateCommandQueueWithProperties(my_context, my_device_id, 0, &err); OPENCL_ERR_CHECK(err);



  //
  /////////////////////////// Part 2: Create a program and extract kernels ///////////////////////////////////
  //

  Timer timer;
  double sum = 0;
  double refsum = 0;
  timer.reset();

  //
  // Build the program:
  //
  size_t source_len = std::string(my_opencl_program).length();
  cl_program prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
  err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

  //
  // Print compiler errors if there was a problem:
  //
  if (err != CL_SUCCESS) {

    char *build_log;
    size_t ret_val_size;
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    build_log = (char *)malloc(sizeof(char) * (ret_val_size+1));
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0'; // terminate string
    std::cout << "Log: " << build_log << std::endl;
    free(build_log);
    std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Extract the only kernel in the program:
  //
  cl_kernel my_kernel = clCreateKernel(prog, "vec_add", &err); OPENCL_ERR_CHECK(err);

  std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;


  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  //
  // Set up buffers on host:
  //
  unsigned int NUMBEROFTHREADS = 4;
  unsigned int NUMBEROFVEC = 5;
  double CPU_result;
  cl_uint vector_size = NUMBEROFTHREADS*NUMBEROFVEC;
  unsigned int N = NUMBEROFTHREADS*NUMBEROFVEC;
  std::vector<ScalarType> x(vector_size, 1.0);
  std::vector<ScalarType> y(vector_size, 1.0);
  std::vector<ScalarType> result(vector_size, 0);
  //
  // Set up buffers for CUDA and CPU kernels on host:
  //
  ScalarType *cuda_X, *cuda_Y;
  ScalarType *X = (ScalarType *)malloc(sizeof(ScalarType) * N);
  ScalarType *Y = (ScalarType *)malloc(sizeof(ScalarType) * N);
  ScalarType *dot = (ScalarType *)malloc(sizeof(ScalarType) * N);
  std::fill(X, X + (N), 1.);
  std::fill(Y, Y + (N), 1.);
  std::fill(dot, dot + (N), 0);

  cudaMemcpy(cuda_X, X, sizeof(ScalarType)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_Y, Y, sizeof(ScalarType)*N, cudaMemcpyHostToDevice);

  std::cout << "Vectorsize: " << vector_size << std::endl;

  timer.reset();

  //CPU_dot(X, Y, dot, N);

  std::cout << "Time to run the dotpord on the CPU: " << timer.get() << std::endl;

  timer.reset();

  //GPU_dot<<<128, 128>>>(cuda_X, cuda_Y, d_tmp, N);

  std::cout << "Time to run the dotpord on the GPU: " << timer.get() << std::endl;

  /*
  std::cout << "x:" << "|";
  for (int i = 0; i < vector_size; i++)
  {
    std::cout << x[i] << "|";
  }
  std::cout  << " " << std::endl;
  std::cout << "y:" << "|";
    for (int i = 0; i < vector_size; i++)
  {
    std::cout  << y[i] << "|";
  }
  */
  std::cout << std::endl;
  std::cout << "Vectors before kernel launch:" << std::endl;

  //
  // Now set up OpenCL buffers:
  //
  cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_result = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(result[0]), &err); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 4: Run kernel ///////////////////////////////////
  //
  size_t  local_size = NUMBEROFTHREADS;
  size_t global_size = NUMBEROFTHREADS*NUMBEROFTHREADS;
  timer.reset();
  //
  // Set kernel arguments:
  //
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_result); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);


  //clSetKernelArg(kernel, 3, array_size*sizeof(int), NULL);

  //
  // Enqueue kernel in command queue:
  //
  err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
  //

  err = clEnqueueReadBuffer(my_queue, ocl_x, CL_TRUE, 0, sizeof(ScalarType) * x.size(), &(x[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);
  err = clEnqueueReadBuffer(my_queue, ocl_result, CL_TRUE, 0, sizeof(ScalarType) * vector_size, &(result[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);
  std::cout << "Time to run and get the data back: " << timer.get() << std::endl;


  std::cout << std::endl;
  std::cout << "Vectors after kernel execution:" << std::endl;
  /*
  std::cout << "x:" << "|";
  for (int i = 0; i < vector_size; i++)
  {
    std::cout << x[i] << "|";
  }
  std::cout  << " " << std::endl;
  std::cout << "y:" << "|";
    for (int i = 0; i < vector_size; i++)
  {
    std::cout  << y[i] << "|";
  }
  std::cout  << " " << std::endl;
  std::cout << "result:" << "|";
  */
    for (int i = 0; i < vector_size; i++)
  {
    sum += result[i];
  }
  std::cout << "dotprod_opencl: " << sum << std::endl;
  std::cout << "dotprod_ref: " << refsum << std::endl;
  std::cout << "dotprod_CPU: " << CPU_result << std::endl;


  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseMemObject(ocl_result);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);

  std::cout << std::endl;
  std::cout << "#" << std::endl;
  std::cout << "# My first OpenCL application finished successfully!" << std::endl;
  std::cout << "#" << std::endl;
  return EXIT_SUCCESS;
}
