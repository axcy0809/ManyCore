#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

#include <vexcl/vexcl.hpp>

typedef double       ScalarType;


int main(void)
{ 
    vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
    std::cout << ctx << std::endl; // print list of selected devices 

    std::fstream csv_times;
    std::string csv_times_name = "dot_VexCL_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;VexCL_time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    ScalarType VexCL_Time = 0;
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

        std::vector<ScalarType> x(N,1);
        std::vector<ScalarType> y(N,2);

        vex::vector<ScalarType> X(ctx, x);
        vex::vector<ScalarType> Y(ctx, y);

        ///////////////////////////////////////  Benchmark start VexCL  //////////////////////////////////////////
        timer.reset();
        vex::Reductor<double, vex::SUM> DOT(ctx);
        for (int i = 0; i < anz; i++)
        {
            dot = DOT((X+Y)*(X-Y));
        }
        VexCL_Time = timer.get()/anz;
        std::cout << "dot-prod_VexCL: " << dot << std::endl;
        std::string sep = ";";
        csv_times << N << sep << VexCL_Time << std::endl;

       
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}

  