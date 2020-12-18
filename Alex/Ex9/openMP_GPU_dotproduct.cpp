#include <omp.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include "timer.hpp"

int main()
{
    std::fstream csv_times;
    std::string csv_times_name = "dot_opemMP_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;openMP_time;CPU_time";
        // to csv file
    csv_times << header << std::endl;
    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    double openMP_time = 0;
    double CPU_time = 0;
    double red = 0;
    double refsum = 0;
    Timer timer;

    for (int i = 1; i < 8; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {

        double *x = (double*)malloc(sizeof(double) * N);
        double *y = (double*)malloc(sizeof(double) * N);

        for (size_t i=0; i<N; ++i) 
        {
            x[i] = 1;
            y[i] = 2;
        }
        
        timer.reset();
        for(int k = 0; k < anz; k++)
        {
            for (size_t i=0; i<N; ++i) 
            {
                refsum += (x[i]+y[i])*(x[i]-y[i]);
            }
        }
        CPU_time = timer.get()/anz;
        std::cout << "CPU_time: " << CPU_time << std::endl;
    

        timer.reset();
        for(int k = 0; k < anz; k++)
        {
            #pragma omp target teams distribute parallel for map(to: x[0:N]) map(to: y[0:N]) map(tofrom: red) reduction(+:red)
            for (int idx = 0; idx < N; ++idx)
            {
                red += (x[idx]+y[idx])*(x[idx]-y[idx]);
            } 
        }
        openMP_time = timer.get()/anz;
        std::cout << "openMP_time: " << openMP_time << std::endl;
        std::cout << "CPU_result: " << refsum/anz << std::endl;
        std::cout << "openMP_result: " << red/anz << std::endl;
        std::string sep = ";";
        csv_times << N << sep << openMP_time << sep << CPU_time << std::endl;
    }
    csv_times.close();
    //csv_diffKernels.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex9/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}