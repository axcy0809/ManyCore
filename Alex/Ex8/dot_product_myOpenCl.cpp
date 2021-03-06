typedef double       ScalarType;


#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>

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
    "   int d = 0;\n"
    "  for (unsigned int i  = get_global_id(0);\n"
    "                    i  < N;\n"
    "                    i += get_global_size(0))\n"
    "{\n"
    "    result[i] = (x[i]+y[i]) * (x[i]-y[i]);\n"
    "}\n"
    "}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.


int main()
{
    std::fstream csv_times;
    std::string csv_times_name = "dot_myopenCL_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;myopenCL_time";
        // to csv file
    csv_times << header << std::endl;
    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;

    for (int i = 1; i < 8; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
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

        //
        /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
        //

        //
        // Set up buffers on host:
        //
        cl_uint vector_size = N;
        std::vector<ScalarType> x(vector_size, 1.0);
        std::vector<ScalarType> y(vector_size, 2.0);
        std::vector<ScalarType> result(vector_size, 0);


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
        double sum = 0;
        double opencl_time = 0;
        std::cout << "N" << N << std::endl;
        for (int j = 0;j < anz;j++)
        {
            //
            // Set kernel arguments:
            //
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


            //
            /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
            //

            err = clEnqueueReadBuffer(my_queue, ocl_x, CL_TRUE, 0, sizeof(ScalarType) * x.size(), &(x[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);
            err = clEnqueueReadBuffer(my_queue, ocl_result, CL_TRUE, 0, sizeof(ScalarType) * vector_size, &(result[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

            sum = 0;
            for (std::size_t i = 0; i < vector_size; i++)
            {
                sum += result[i];
            }
        }    
        opencl_time = timer.get()/anz;
        std::cout << "opencl dotprod: " << sum << std::endl;
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;
        std::string sep = ";";
        csv_times << N << sep << opencl_time << std::endl;
        //
        // cleanup
        //
        clReleaseMemObject(ocl_x);
        clReleaseMemObject(ocl_y);
        clReleaseMemObject(ocl_result);
        clReleaseProgram(prog);
        clReleaseCommandQueue(my_queue);
        clReleaseContext(my_context);

    }

    std::cout << std::endl;
    std::cout << "#" << std::endl;
    std::cout << "# My first OpenCL application finished successfully!" << std::endl;
    std::cout << "#" << std::endl;
    csv_times.close();
    //csv_diffKernels.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;
    return EXIT_SUCCESS;
    }
