#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#define VIENNACL_WITH_CUDA
typedef double       ScalarType;


int main(void)
{ 
    std::fstream csv_times;
    std::string csv_times_name = "dot_ViennaCl_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;ViennaCl_time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    ScalarType ViennaCl_Time = 0;
    Timer timer;
    ScalarType dot = 0;

    for (int i = 1; i < 8; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {
        std::cout << "N: " << N << std::endl;
        ///////////////////////////////////////  Benchmark start on the ViennaCl as additional reference  //////////////////////////////////////////

        viennacl::vector<double> x_VIE = viennacl::scalar_vector<double>(N, 1.0);
        viennacl::vector<double> y_VIE = viennacl::scalar_vector<double>(N, 2.0);
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            dot = viennacl::linalg::inner_prod(x_VIE + y_VIE,x_VIE - y_VIE);
        }
        ViennaCl_Time = timer.get()/anz;

        std::cout << "dot-prod_ViennaCL: " << dot << std::endl;

        ///////////////////////////////////////  create the outputfile //////////////////////////////////////////
        std::string sep = ";";
        csv_times << N << sep << ViennaCl_Time << std::endl;
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}

  