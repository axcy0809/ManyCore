#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

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

int main(void)
{ 
    std::fstream csv_times;
    std::string csv_times_name = "dot_CPU_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;CPU_time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    ScalarType CPU_Time = 0;
    Timer timer;
    ScalarType reference = 0;

    for (int i = 1; i < 8; i++)
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

        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            reference = vec_dot(vecplus(x,y,1),vecplus(x,y,-1));
        }
        CPU_Time = timer.get()/anz;
        std::cout << "reference result from CPU: " << reference << std::endl;

        ///////////////////////////////////////  create the outputfile //////////////////////////////////////////
        std::string sep = ";";
        csv_times << N << sep << CPU_Time << std::endl;
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}

  