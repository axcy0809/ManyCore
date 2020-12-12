#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>

#include <vexcl/vexcl.hpp>
     
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/inner_product.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>


#include <cstdlib>

#define BLOCK_SIZE 256
#define VIENNACL_WITH_CUDA
typedef double       ScalarType;


std::vector<ScalarType> vecplus(std::vector<ScalarType> x, std::vector<ScalarType> y, int sign)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] + sign * y[i];
    }
    return x;
}

ScalarType vec_dot(std::vector<ScalarType> x, std::vector<ScalarType> y)
{
    double sum = 0;
    for (int i = 0; i < x.size(); i++)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

namespace compute = boost::compute;

int main(void)
{ 
    vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
    std::cout << ctx << std::endl; // print list of selected devices 

    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);


    std::fstream csv_times;
    std::string csv_times_name = "dot_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;CPU_time;BoostCompute_Time;Thrust_Time;VexCL_Time;ViennaCL_Time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;

    ScalarType CPU_Time = 0;
    ScalarType BoostCompute_Time = 0;
    ScalarType Thrust_Time = 0;
    ScalarType VexCL_Time = 0;
    ScalarType ViennaCL_Time = 0;


    Timer timer;
    ScalarType reference = 0;
    for (int i = 1; i < 2; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {
        std::cout << "N: " << N << std::endl;
    ///////////////////////////////////////  Benchmark start on the CPU as additional reference  //////////////////////////////////////////

        std::vector<ScalarType> x(N,1);
        std::vector<ScalarType> y(N,2);
        std::vector<ScalarType> init(N,0);

        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            reference = vec_dot(vecplus(x,y,1),vecplus(x,y,-1));
        }
        CPU_Time = timer.get()/anz;

    ///////////////////////////////////////  declaration for the VexCL  //////////////////////////////////////////
        vex::vector<ScalarType> X(ctx, x);
        vex::vector<ScalarType> Y(ctx, y);

        double s_Boost = 0;

    //////////////////////////////////////declaration for thrust  ////////////////////////////////////////////////
    std::vector<double> v1(N,1);
    std::vector<double> v2(N,2);
    std::vector<double> v1pv2(N,1);
    std::vector<double> v1mv2(N,2);

    
    for (int i = 0; i < N; i++)
    {
        v1pv2[i] = v1[i] + v2[i];
        v1mv2[i] = v1[i] - v2[i];
    }
    

    std::cout << "v1pv2: " << v1pv2[0] << std::endl;
    std::cout << "v1mv2: " << v1mv2[0] << std::endl;

    thrust::host_vector<double> h_v1 = v1pv2;
    thrust::host_vector<double> h_v2 = v1mv2;

    thrust::device_vector<ScalarType> d_v1 = h_v1;
    thrust::device_vector<ScalarType> d_v2 = h_v2;

    ScalarType start = 0;

    ScalarType expected_thrust = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), start);
    //std::cout << "host dot: " << expected_thrust << std::endl;

    /*
        thrust::host_vector<ScalarType> x_t(N,1);
        thrust::host_vector<ScalarType> y_t(N,2);

        //x_t = x_t+y_t;
        //y_t = x_t-y_t;

        thrust::device_vector<ScalarType> d_x_t= x_t;
        thrust::device_vector<ScalarType> d_y_t= y_t;

        double start = 0;

        double expected = thrust::inner_product(d_x_t.begin(), d_x_t.end(), d_y_t.begin(), start);

        //double result_thrust = thrust::inner_product(thrust::ho, 0, d_x_t+d_y_t, d_x_t-d_y_t, 0.0f);
    
        thrust::host_vector<double> h_v1(32 << 20);
        thrust::host_vector<double> h_v2(32 << 20);
        thrust::generate(h_v1.begin(), h_v1.end(), rand);
        thrust::generate(h_v2.begin(), h_v2.end(), rand);

        thrust::device_vector<ScalarType> d_v1 = h_v1;
        thrust::device_vector<ScalarType> d_v2 = h_v2;

        ScalarType start = 0;

        //ScalarType expected_thrust = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), start);
        ScalarType expected_thrust = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), start);
        std::cout << "host dot: " << expected_thrust << std::endl;
    /*
        thrust::generate(h_v1.begin(), h_v1.end(), rand);
        thrust::generate(h_v2.begin(), h_v2.end(), rand);

        thrust::device_vector<ScalarType> d_v1 = h_v1;
        thrust::device_vector<ScalarType> d_v2 = h_v2;

        ScalarType start = 13;

        ScalarType expected_thrust = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), start);
        ScalarType result_thrust   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), start);

        std::cout << "thrust dot: " << expected_thrust << std::endl;
        std::cout << "host dot: " << result_thrust << std::endl;
    */
    ///////////////////////////////////////  declaration for the Boost  //////////////////////////////////////////

        // create a vector on the device
        compute::vector<ScalarType> d_x(x.size(), context);
        compute::vector<ScalarType> d_y(y.size(), context);
        compute::vector<ScalarType> d_xply(y.size(), context);
        compute::vector<ScalarType> d_xmiy(y.size(), context);

        // transfer data from the host to the device
        compute::copy(
            x.begin(), x.end(), d_x.begin(), queue
        );
        compute::copy(
            y.begin(), y.end(), d_y.begin(), queue
        );
        compute::copy(
            init.begin(), init.end(), d_xply.begin(), queue
        );
        compute::copy(
            init.begin(), init.end(), d_xmiy.begin(), queue
        );
        for (int i = 0; i < anz; i++)
        {
            compute::transform(d_x.begin(), d_x.end(), 
                d_y.begin(), d_xply.begin(), compute::plus<double>{}, queue);

            compute::transform(d_x.begin(), d_x.end(), 
                d_y.begin(), d_xmiy.begin(), compute::minus<double>{}, queue);

            s_Boost = compute::inner_product(d_xply.begin(), d_xply.end(), 
                        d_xmiy.begin(), 0.0, queue);
        }
        BoostCompute_Time = timer.get()/anz;

        double s = 0;

    /////////////////////////////////////declaration for VIENNA   //////////////////////////////////////////////////

        double s_VIE = 0;

        viennacl::vector<double> x_VIE = viennacl::scalar_vector<double>(N, 1.0);
        viennacl::vector<double> y_VIE = viennacl::scalar_vector<double>(N, 2.0);

        for (int i = 0; i < anz; i++)
        {
            s_VIE = viennacl::linalg::inner_prod(x_VIE + y_VIE,x_VIE - y_VIE);
        }
        ViennaCL_Time = timer.get()/anz;

    ///////////////////////////////////////  Benchmark start VexCL  //////////////////////////////////////////
        timer.reset();
        vex::Reductor<double, vex::SUM> DOT(ctx);
        for (int i = 0; i < anz; i++)
        {
            s = DOT((X+Y)*(X-Y));
        }
        VexCL_Time = timer.get()/anz;

    ///////////////////////////////////////  cout comands  //////////////////////////////////////////////////

        std::cout << "reference result from CPU: " << reference << std::endl;
        std::cout << "dot-prod_VexCL: " << s << std::endl;
        std::cout << "dot-prod_ViennaCL: " << s_VIE << std::endl;
        std::cout << "dot-prod_Boost: " << s_Boost << std::endl;
        std::cout << std::endl;

     ///////////////////////////////////////  create the outputfile //////////////////////////////////////////
        std::string sep = ";";
        csv_times << N << sep << CPU_Time << sep << BoostCompute_Time << sep << Thrust_Time << sep << VexCL_Time << sep << ViennaCL_Time << std::endl;
    }
    csv_times.close();
    //csv_diffKernels.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;


  return EXIT_SUCCESS;
}
