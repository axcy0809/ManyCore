    //
    // Tutorial for demonstrating a simple OpenCL vector addition kernel
    //
    // Author: Karl Rupp    rupp@iue.tuwien.ac.at
    //

    typedef double       ScalarType;

    #include <fstream>
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


    const char *my_opencl_program = ""
    "__kernel void vec_mult(__global double *x,\n"
    "                      __global double *y,\n"
    "                      __global double *result,\n"
    "                      unsigned int N\n)"
    "{\n"
    "  for (unsigned int i  = get_global_id(0);\n"
    "                    i  < N;\n"
    "                    i += get_global_size(0))\n"
    "    result[i] = x[i] * y[i];\n"
    "}";   // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.

    void CPU_dot(ScalarType *x, ScalarType *y, ScalarType *result, unsigned int N)
    {
        for (int i = 0; i < N; i++)
        {
            result[i] = x[i] * y[i];
        }

    }

    __global__ void GPU_dot (double *x, double *y, double *dot, unsigned int N)
{
    unsigned int ind = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int str = blockDim.x*gridDim.x;

    double tmpsum = 0.0;
    while(ind < N)
    {
        tmpsum = x[ind]*y[ind];
        ind += str;
    }

    dot[threadIdx.x] = tmpsum;
}


    int main()
    {
    std::vector<unsigned int> vec_Ns{100, 1000};

    std::fstream csv_times;
    std::string csv_times_name = "dot_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;CPU_time;opencl_time;factor";
        // to csv file
    csv_times << header << std::endl;

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
    cl_kernel my_kernel = clCreateKernel(prog, "vec_mult", &err); OPENCL_ERR_CHECK(err);

    std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;


    //
    /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
    //

    //
    // Set up buffers on host:
    //
    unsigned int H = 10;
    double CPU_time = 0;
    double GPU_time = 0;
    double opencl_time = 0;
    unsigned int K = H;
    int anz = 100;
    cl_uint vector_size = K;
    unsigned int N = K;
    double refsum = 0;
    std::vector<ScalarType> x(vector_size, 2.0);
    std::vector<ScalarType> y(vector_size, 2.0);
    std::vector<ScalarType> result(vector_size, 0);

    ScalarType *X, *Y, *dot_CPU, *dot_GPU;
    ScalarType *cuda_X, *cuda_Y, *cuda_dot_GPU;
    X = (ScalarType*)malloc(sizeof(ScalarType) * N);
    Y = (ScalarType*)malloc(sizeof(ScalarType) * N);
    dot_CPU = (ScalarType*)malloc(sizeof(ScalarType) * N);
    dot_GPU = (ScalarType*)malloc(sizeof(ScalarType) * 128);
    std::cout << "Vectorsize: " << N << std::endl;

    cudaMalloc(&cuda_X, N*sizeof(double));
    cudaMalloc(&cuda_Y, N*sizeof(double));
    cudaMalloc(&cuda_dot_GPU, 128*sizeof(double));

    for(std::size_t i = 0; i < vector_size; i++)
    {
        X[i] = 2.;
        Y[i] = 2.;
        dot_CPU[i] = 0;
        refsum += x[i] * y[i];
    }

    std::cout << std::endl;
    std::cout << "Vectors before kernel launch:" << std::endl;
    std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << " ..." << std::endl;
    std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;

    cudaMemcpy(cuda_X, X, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Y, Y, N*sizeof(double), cudaMemcpyHostToDevice);
    std::fill(dot_GPU, dot_GPU + (128), 0);

    timer.reset();
    for (int i = 0;i < anz; i++)
    {
            CPU_dot(X, Y, dot_CPU, N);
    }
    CPU_time = timer.get();
    std::cout << "Time for CPU kernel: " << CPU_time/anz << std::endl;

    timer.reset();
    for (int i = 0;i < anz; i++)
    {
        GPU_dot<<<128, 128>>>(cuda_X, cuda_Y, cuda_dot_GPU, N);
        cudaDeviceSynchronize();
        cudaMemcpy(dot_GPU, cuda_dot_GPU, sizeof(double), cudaMemcpyDeviceToHost);
    }
    //cudaMemcpy(dot_GPU, cuda_dot_GPU, sizeof(double), cudaMemcpyDeviceToHost);
    GPU_time = timer.get();
    std::cout << "Time for GPU kernel: " << GPU_time/anz << std::endl;

    //
    // Now set up OpenCL buffers:
    //
    cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
    cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
    cl_mem ocl_result = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(result[0]), &err); OPENCL_ERR_CHECK(err);


    //
    /////////////////////////// Part 4: Run kernel ///////////////////////////////////
    //
    size_t  local_size = 128;
    size_t global_size = 128*128;
    for (int i = 0;i < anz; i++)
    {
        //
        // Set kernel arguments:
        //
        timer.reset();
        err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_result); OPENCL_ERR_CHECK(err);
        err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);

        //
        // Enqueue kernel in command queue:
        //
        err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

        // wait for all operations in queue to finish:
        err = clFinish(my_queue); OPENCL_ERR_CHECK(err);
    }
    opencl_time = timer.get();
    std::cout << "Time for opencl kernel: " << opencl_time/anz << std::endl;
    

    std::cout << "opencl is " << CPU_time/opencl_time << " faster than CPU" << std::endl;
    std::cout << "CUDA is " << CPU_time/GPU_time << " faster than GPU" << std::endl;



    //
    /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
    //

    err = clEnqueueReadBuffer(my_queue, ocl_x, CL_TRUE, 0, sizeof(ScalarType) * x.size(), &(x[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);
    err = clEnqueueReadBuffer(my_queue, ocl_result, CL_TRUE, 0, sizeof(ScalarType) * vector_size, &(result[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);


    std::cout << std::endl;
    std::cout << "Vectors after kernel execution:" << std::endl;
    std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << " ..." << std::endl;
    std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << " ..." << std::endl;
    //
    // cleanup
    //
    double sum = 0;
    double CPU_sum = 0;
    double GPU_sum = 0;
    for (std::size_t i = 0; i < vector_size; i++)
    {
        sum += result[i];
        CPU_sum += dot_CPU[i];
    }
    for (int i = 0; i < 10; i++)
    {
        GPU_sum = GPU_sum + dot_GPU[i];
        //std::cout << "dot: " << GPU_sum << "einz: " << dot_CPU[i] << std::endl;  
        //std::cout << dot_GPU[i] + dot_GPU[i]  << std::endl;     
    }
    std::cout << "GPU dotprod: " << GPU_sum << std::endl;
    std::cout << "CPU dotprod: " << CPU_sum << std::endl;
    std::cout << "opencl dotprod: " << sum << std::endl;
    std::cout << "reference dotprod: " << refsum << std::endl;
    std::cout << "GPU dotprod: " << dot_CPU[0] << std::endl;
    std::cout << "GPU dotprod: " << dot_CPU[1] << std::endl;
    std::cout << "GPU dotprod: " << dot_CPU[2] << std::endl;
    //
    // define outputstream
    //
    std::string sep = ";";
    csv_times << N << sep << CPU_time << sep << opencl_time << std::endl;

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
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex7/" + csv_times_name << std::endl;
    return EXIT_SUCCESS;
    }
