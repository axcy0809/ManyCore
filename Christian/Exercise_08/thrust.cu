#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <algorithm>

#include "timer.hpp"


int main(void) {

    int N = 1000;
    Timer timer;

  // initialize x and y
  thrust::host_vector<double> h_x(N, 1.0);
  thrust::host_vector<double> h_y(N, 2.0);


  // transfer data to the device
  thrust::device_vector<double> d_x = h_x;
  thrust::device_vector<double> d_y = h_y;
  thrust::device_vector<double> d_arg1(N);
  thrust::device_vector<double> d_arg2(N);

  std::vector<double> timings;
  double z;
  for(int reps=0; reps < 10; ++reps) {
      timer.reset();
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg1.begin(), thrust::plus<double>());
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_arg2.begin(), thrust::minus<double>());

    z = thrust::inner_product(d_arg1.begin(), d_arg1.end(), d_arg2.begin(), 0.0);
    timings.push_back(timer.get());        
  }
std::sort(timings.begin(), timings.end());
double time_elapsed = timings[10/2];

std::cout << "Time elapsed: " << time_elapsed << std::endl << std::endl;

  std::cout << "Inner Product = " << z << std::endl;

  return 0;
}
