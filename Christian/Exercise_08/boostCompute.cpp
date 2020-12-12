#include <vector>
#include <algorithm>
#include <iostream>
 
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
#include <boost/geometry/arithmetic/dot_product.hpp>
#include <boost/range/numeric.hpp>
#include "timer.hpp"

 
namespace compute = boost::compute;
 
int main()
{
    int N = 1000;
    Timer timer;

    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);
 
    // generate random data on the host
    std::vector<double> x(N, 1);
    std::vector<double> y(N, 2);
 
   // create a vector on the device
   compute::vector<double> d_x(x.size(), context);
   compute::vector<double> d_y(y.size(), context);
   compute::vector<double> d_arg1(N, context);
   compute::vector<double> d_arg2(N, context);
 
   // transfer data from the host to the device
   compute::copy(x.begin(), x.end(), d_x.begin(), queue);
   compute::copy(y.begin(), y.end(), d_y.begin(), queue);
    // calculate inner product

std::vector<double> timings;
  double z;
  for(int reps=0; reps < 10; ++reps) {
      timer.reset();
    compute::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg1.begin(), compute::plus<double>{}, queue);
    compute::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg2.begin(), compute::minus<double>{}, queue);
     z = boost::inner_product(d_arg1, d_arg2, 0);
     timings.push_back(timer.get());        
    }
  std::sort(timings.begin(), timings.end());
  double time_elapsed = timings[10/2];

  std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;
    
    std::cout << "Dot Product = " << z << std::endl; 
 
    return 0;
}