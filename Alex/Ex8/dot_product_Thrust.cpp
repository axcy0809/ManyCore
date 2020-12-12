#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>

#include <cstdlib>

typedef double       ScalarType;


int main(void)
{ 
    std::fstream csv_times;
    std::string csv_times_name = "dot_Thrust_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;Thrust_time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    ScalarType VexCL_Time = 0;
    Timer timer;

    for (int i = 1; i < 8; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {
        ScalarType dot = 0;
        ScalarType start = 0;
        std::cout << "N: " << N << std::endl;
        ///////////////////////////////////////  Benchmark start on the CPU as additional reference  //////////////////////////////////////////

        std::vector<ScalarType> x(N,1);
        std::vector<ScalarType> y(N,2);
        std::vector<ScalarType> v1pv2(N,0);
        std::vector<ScalarType> v1mv2(N,0);

        for (int i = 0; i < N; i++)
        {
            v1pv2[i] = x[i] + y[i];
            v1mv2[i] = x[i] - y[i];
        }
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            thrust::host_vector<ScalarType> h_v1 = v1pv2;
            thrust::host_vector<ScalarType> h_v2 = v1mv2;

            thrust::device_vector<ScalarType> d_v1 = h_v1;
            thrust::device_vector<ScalarType> d_v2 = h_v2;

            dot = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), start);
        }
        VexCL_Time = timer.get()/anz;
        std::cout << "dot-prod_Thrust: " << dot << std::endl;
    ///////////////////////////////////////  create the outputfile //////////////////////////////////////////
        std::string sep = ";";
        csv_times << N << sep << VexCL_Time << std::endl;
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}

  