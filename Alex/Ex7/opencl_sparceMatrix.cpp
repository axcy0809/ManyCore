
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "generate.hpp"
#include "timer.hpp"
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "ocl-error.hpp"

typedef double       ScalarType;

const char *my_opencl_program = ""
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
""
"__kernel void csr_matvec_product_opencl(unsigned int N,\n"
"                                        __global int *csr_rowoffsets,\n"
"                                        __global int  *csr_colindices,\n"
"                                        __global double *csr_values,\n"
"                                        __global double *x,\n"
"                                        __global double *y)\n"
"{\n"
"  for (unsigned int i  = get_global_id(0);\n"
"                    i  < N;\n"
"                    i += get_global_size(0))\n"
"   {\n"
"           double value = 0;\n"
"           for (int j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)\n"
"           value += csr_values[j] * x[csr_colindices[j]];\n"
"                                                         \n"
"           y[i] = value;\n"
"    }\n"
"}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.


/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y. CPU implementation.  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double const *x, double *y)
{
  for (size_t i=0; i<N; ++i) {
    double value = 0;
    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)
      value += csr_values[j] * x[csr_colindices[j]];

    y[i] = value;
  }

}

void diff_in_solution(std::vector<ScalarType> y, double *y_ref, int N)
{
  int count = 0;
  for (int i = 0; i < N; i++)
  {
    if(y[i] == y_ref[i])
    {
      count = count + 1;
    }   
  }
  if(count == N)
  {
    printf("Right Solution");
  }
  else
  {
    printf("Wrong Solution");
  }
  
}


/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void benchmark_matvec(size_t points_per_direction, size_t max_nonzeros_per_row,
                      void (*generate_matrix)(size_t, int*, int*, double*)) // function pointer parameter
{

  size_t N = points_per_direction * points_per_direction; // number of rows and columns

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * max_nonzeros_per_row * N);
  double *csr_values  = (double*)malloc(sizeof(double) * max_nonzeros_per_row * N);

  //
  // fill CSR matrix with values
  //
  generate_matrix(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  //
  // Allocate vectors:
  //
  double *x = (double*)malloc(sizeof(double) * N); std::fill(x, x + N, 1);
  double *y = (double*)malloc(sizeof(double) * N); std::fill(y, y + N, 0);


  //
  // Call matrix-vector product kernel
  //
  Timer timer;
  timer.reset();
  csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, x, y);
  std::cout << "Time for product: " << timer.get() << std::endl;

  free(x);
  free(y);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main() 
{
  cl_int err;

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
  cl_kernel my_kernel = clCreateKernel(prog, "csr_matvec_product_opencl", &err); OPENCL_ERR_CHECK(err);

  std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;

  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  int points_per_direction = 4;
  int max_nonzeros_per_row = 4;
  size_t N = points_per_direction * points_per_direction; // number of rows and columns

  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * max_nonzeros_per_row * N);
  double *csr_values  = (double*)malloc(sizeof(double) * max_nonzeros_per_row * N);

  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);
  std::cout << "Output from the set matrix laplace: " << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout << "i: " << i << " | " << csr_values[i] << std::endl;
  }
  //
  // Allocate vectors:
  //
  double *x_ref = (double*)malloc(sizeof(double) * N); std::fill(x_ref, x_ref + N, 1);
  double *y_ref = (double*)malloc(sizeof(double) * N); std::fill(y_ref, y_ref + N, 0);

  //double *x = (double*)malloc(sizeof(double) * N); std::fill(x, x + N, 1);
  //double *y = (double*)malloc(sizeof(double) * N); std::fill(y, y + N, 0);

  std::vector<ScalarType> x(N, 1.0);
  std::vector<ScalarType> y(N, 0.0);

  csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, x_ref, y_ref);
  std::cout << "Solution: " << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout << "i: " << i << " | " << y_ref[i] << std::endl;
  }

  //
  // Set up buffers on host:
  //

  cl_uint vector_size = N;

  //
  // Now set up OpenCL buffers:
  //
  cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_csr_rowoffsets = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (N+1) * sizeof(int), &(csr_rowoffsets[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_csr_colindices = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_nonzeros_per_row * N * sizeof(int), &(csr_colindices[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_csr_values = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_nonzeros_per_row * N * sizeof(ScalarType), &(csr_values[0]), &err); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 4: Run kernel ///////////////////////////////////
  //
  size_t  local_size = 128;
  size_t global_size = 128*128;

  //
  // Set kernel arguments:
  //
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem), (void*)&ocl_csr_rowoffsets); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_csr_colindices); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_mem),  (void*)&ocl_csr_values); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 4, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 5, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);

  //
  // Enqueue kernel in command queue:
  //
  err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
  //

  err = clEnqueueReadBuffer(my_queue, ocl_y, CL_TRUE, 0, sizeof(ScalarType) * y.size(), &(y[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  std::cout << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout << "i: " << i << " | " << y[i] << std::endl;
  }
  std::cout << std::endl;
  std::cout << std::endl;
  diff_in_solution(y,y_ref,N);

  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);










  



  /*
  std::cout << "# Benchmarking finite difference matrix" << std::endl;
  benchmark_matvec(100, 5, generate_fdm_laplace); // 100*100 unknowns, finite difference matrix

  std::cout << "# Benchmarking special matrix" << std::endl;
  benchmark_matvec(100, 2000, generate_matrix2);     // 100*100 unknowns, special matrix with 200-2000 nonzeros per row
  */
  free(x_ref);
  free(y_ref);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);

  return EXIT_SUCCESS;
}