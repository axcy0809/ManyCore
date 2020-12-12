#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/inner_product.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

typedef double       ScalarType;
namespace compute = boost::compute;

int main(void)
{ 
    //vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
    //std::cout << ctx << std::endl; // print list of selected devices 

    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    std::fstream csv_times;
    std::string csv_times_name = "dot_Boost_times.csv";
    csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
    std::string header = "N;Boost_time";
        // to csv file
    csv_times << header << std::endl;

    std::vector<int> vec_Ns;
    int hoch = 1;
    int anz = 20;
    ScalarType Boost_Time = 0;
    Timer timer;
    ScalarType dot = 0;

    for (int i = 1; i < 2; i++)
    {
        hoch = hoch * 10;
        vec_Ns.push_back(hoch);
    }
    for (int &N : vec_Ns)
    {
        std::cout << "N: " << N << std::endl;

        std::vector<ScalarType> x(N,1);
        std::vector<ScalarType> y(N,2);
        std::vector<ScalarType> init(N,0);

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
        timer.reset();
        for (int i = 0; i < anz; i++)
        {
            compute::transform(d_x.begin(), d_x.end(), 
                d_y.begin(), d_xply.begin(), compute::plus<double>{}, queue);

            compute::transform(d_x.begin(), d_x.end(), 
                d_y.begin(), d_xmiy.begin(), compute::minus<double>{}, queue);

            dot = compute::inner_product(d_xply.begin(), d_xply.end(), 
                        d_xmiy.begin(), 0.0, queue);
        }
        Boost_Time = timer.get()/anz;
        std::cout << "dot-prod_Boost: " << dot << std::endl;
        ///////////////////////////////////////  create the outputfile //////////////////////////////////////////

        std::string sep = ";";
        csv_times << N << sep << Boost_Time << std::endl;
    }
    csv_times.close();
    std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex8/" + csv_times_name << std::endl;

    return EXIT_SUCCESS;
}

  